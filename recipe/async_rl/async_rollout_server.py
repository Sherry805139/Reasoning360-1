"""
Customized verl server and workers to support weight update.
This creates the interruptable worker and the corresponding client (AsyncLLM Server).
TODO: if allows, we should just uses the original class, and directly calls
    'self.engine.engine_core._call_utility_async("update_weights", *args)'
"""

import asyncio
import os
from typing import Any, Callable, Optional

import numpy as np
from omegaconf import DictConfig
import ray
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from tensordict import TensorDict
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.worker.worker_base import WorkerWrapperBase

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    ExternalZeroMQDistributedExecutor,
    ExternalRayDistributedExecutor,
)
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs

from async_rl.vllm_patch import InterruptableAsyncLLM


def _get_worker_dict_name_hint(prefix: str) -> Callable[[str], bool]:
    def _worker_dict_name_hint(actor_name: str) -> bool:
        return actor_name.startswith(f"{prefix}WorkerDict")

    return _worker_dict_name_hint


def _get_separated_worker_name_hint(prefix: str, worker_name: str) -> Callable[[str], bool]:
    def _separated_worker_name_hint(actor_name: str) -> bool:
        return actor_name.startswith(f"{prefix}{worker_name}")

    return _separated_worker_name_hint


def _get_model_runner_workers(vllm_config, init_ray: bool = True, name_hint: Callable[[str], bool] = None):
    """
    NOTE: this is a copy of `_get_model_runner_workers` in vllm_async_server.py.
    The original file used a fixed name hint (WorkerDict) to find actors, which is not compatible with the disaggregated rollout-training scenario.
    """
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
    assert name_hint is not None, "name_hint must be provided for async_rl"

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if name_hint(actor_name)
    ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: list[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class InterruptableAsyncvLLMServer(AsyncServerBase):
    """
    NOTE: we made a copy of verl's AsyncvLLMServer because it is wrapped
    by ray.remote, and thus cannot be inherited.
    This server adds:
    - update_weights API for workers to update weights. (TODO: build connection with the trainer and update by GPU RDMA)
    - link to a data buffer, so that the generated results are sent to the data buffer.
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: InterruptableAsyncLLM = None
        # Instrumentation counters
        self.total_requests_dispatched_train: int = 0
        self.total_requests_dispatched_val: int = 0
        self.total_requests_completed_train: int = 0
        self.total_requests_completed_val: int = 0
        self.total_requests_preempted: int = 0

        # Async RL related fields
        self.dst_data_buffer = None  # Training data buffer
        self.val_dst_data_buffer = None  # Validation data buffer
        sampling_kwargs = dict(
            temperature=self.config.rollout.temperature,
            top_p=self.config.rollout.top_p,
            repetition_penalty=1.0,
            n=1, # NOTE: data preprocessing already repeats the prompts.
            logprobs=1,
            detokenize=False,
        )
        self.sampling_kwargs = sampling_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config = get_generation_config(self.config.model.path)

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        #### Customization on top of verl
        assert not config.free_cache_engine, "async rl never needs to sleep the rollout worker, so free_cache_engine should be false"
        worker_cls = "async_rl.vllm_patch.MutableWeightRayWorker"
        scheduler_cls = "async_rl.vllm_patch.InterruptableScheduler"

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            # NOTE: customized by us
            worker_cls=worker_cls,
            scheduler_cls=scheduler_cls,
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = InterruptableAsyncLLM.from_vllm_config(vllm_config)

        # NOTE: no need for serving chat in async_rl
        # # build serving chat
        # model_config = self.engine.model_config
        # BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        # models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        # self.openai_serving_chat = OpenAIServingChat(
        #     self.engine,
        #     model_config,
        #     models,
        #     "assistant",
        #     request_logger=RequestLogger(max_log_len=4096),
        #     chat_template=None,
        #     chat_template_content_format="auto",
        #     enable_auto_tools=config.multi_turn.tool_config_path is not None,
        #     tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        # )

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # TODO: get it somewhere.
        worker_cls_name = "AsyncRolloutWorker"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False, name_hint=_get_separated_worker_name_hint(self.wg_prefix, worker_cls_name))
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        raise RuntimeError("this API should not be called.")
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        return final_res.outputs[0].token_ids

    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        if self.config.rollout.free_cache_engine:
            await self.engine.sleep()

    ######## NOTE: New function for weight update
    async def update_weights(self, ckpt_path: str):
        preempted = await self.engine.update_weights(ckpt_path)
        if isinstance(preempted, int):
            self.total_requests_preempted += preempted

    async def get_counters(self):
        return {
            "train_sent": self.total_requests_dispatched_train,
            "val_sent": self.total_requests_dispatched_val,
            "train_completed": self.total_requests_completed_train,
            "val_completed": self.total_requests_completed_val,
            "preempted": self.total_requests_preempted,
        }

    ######## NOTE: New function for data buffer
    async def link_data_buffer(self, dst_data_buffer):
        self.dst_data_buffer = dst_data_buffer

    async def link_validation_buffer(self, val_dst_data_buffer):
        self.val_dst_data_buffer = val_dst_data_buffer

    def _split_prompts_to_gen_and_meta(self, batch: DataProto):
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        return gen_batch, batch

    def _pre_process_inputs(self, prompts: DataProto):
        prompts, out_batch = self._split_prompts_to_gen_and_meta(prompts)
        # 1. Handle sampling params
        sampling_params = dict(self.sampling_kwargs)
        if prompts.meta_info.get("validate", False):
            sampling_params["top_p"] = self.config.rollout.val_kwargs.top_p
            sampling_params["temperature"] = self.config.rollout.val_kwargs.temperature

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # Below are copied from vllm_rollout_spmd.py: generate_sequences
        # 2. Handle prompts: non_tensor_batch
        # TODO: learn from AgentLoopWorker.generate_sequences
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sampling_params.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            })
        elif is_validate:
            sampling_params.update({
                "top_k": self.config.rollout.val_kwargs.top_k,
                "top_p": self.config.rollout.val_kwargs.top_p,
                "temperature": self.config.rollout.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            })

        # TODO: handle lora_requests
        return vllm_inputs, sampling_params, prompts, out_batch

    def _post_process_outputs(self, output: RequestOutput, gen_batch: DataProto, out_batch: DataProto):
        """
        This function handles the output-prompt concatenation, as well as the padding.
        """
        # TODO: each prompt data must be exactly matched with one request, so no need
        # to have two dimensions.
        # split gen batch keys out.

        response = []
        rollout_log_probs = []
        for sample_id in range(len(output.outputs)):
            response_ids = output.outputs[sample_id].token_ids
            response.append(response_ids)
            # handle log probs
            curr_log_prob = []
            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                curr_log_prob.append(logprob[response_ids[i]].logprob)
            rollout_log_probs.append(curr_log_prob)
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.max_model_len)

        rollout_log_probs = pad_2d_list_to_length(
            rollout_log_probs, -1, max_length=self.max_model_len
        )
        rollout_log_probs = rollout_log_probs.to(torch.float32)

        idx = gen_batch.batch["input_ids"]
        batch_size = idx.size(0)
        attention_mask = gen_batch.batch["attention_mask"]
        position_ids = gen_batch.batch["position_ids"]
        eos_token_id = gen_batch.meta_info["eos_token_id"]
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        batch_dict = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rollout_log_probs": rollout_log_probs,
            },
            batch_size=batch_size,
        )
        res = DataProto(batch=batch_dict, non_tensor_batch=gen_batch.non_tensor_batch)
        out_batch = out_batch.union(res)

        # NOTE: code copied from ray_trainer.py: fit
        if "response_mask" not in out_batch.batch:
            out_batch.batch["response_mask"] = compute_response_mask(out_batch)
        out_batch.meta_info["global_token_num"] = torch.sum(out_batch.batch["attention_mask"], dim=-1).tolist()
        return out_batch

    async def _run_request(self, prompt: TokensPrompt, sampling_params: SamplingParams, request_id: str, gen_batch: DataProto, out_batch: DataProto, is_validation: bool = False):
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
        output_data = self._post_process_outputs(final_res, gen_batch, out_batch)

        # Add required meta_info fields for the actor
        output_data.meta_info["temperature"] = sampling_params.temperature

        micro_batch_size = self.config.rollout.log_prob_micro_batch_size_per_gpu
        if micro_batch_size is None:
            micro_batch_size = 1
        output_data.meta_info["micro_batch_size"] = micro_batch_size
        
        max_token_len = self.config.rollout.log_prob_max_token_len_per_gpu
        if max_token_len is None:
            max_token_len = 32768
        output_data.meta_info["max_token_len"] = max_token_len

        use_dynamic_bsz = self.config.rollout.log_prob_use_dynamic_bsz
        if use_dynamic_bsz is None:
            use_dynamic_bsz = False
        output_data.meta_info["use_dynamic_bsz"] = use_dynamic_bsz

        if is_validation:
            if self.val_dst_data_buffer is None:
                raise RuntimeError("Validation buffer not linked. Call link_validation_buffer() first.")
            await self.val_dst_data_buffer.add_data.remote(output_data)
        else:
            await self.dst_data_buffer.add_data.remote(output_data)

    async def generate_and_send(self, reqs: DataProto, batch_id: str, is_validation: bool = False):
        """
        This function handles:
        - batch (DataProto) to generate requests (prompt_ids, sampling_params, request_id)
        - calling the engine's generate API
        - post-processing outputs (RequestOoutput) to DataProto and send to the dst data buffer.
        """
        # FIXME: apply chat template somewhere?
        # NOTE: the reqs may be directly pulled from the server, instead of sending from the
        # single controller. This simplifies the network traffic.
        vllm_inputs, sampling_kwargs, gen_batch, out_batch = self._pre_process_inputs(reqs)
        dispatched = len(vllm_inputs)
        if is_validation:
            self.total_requests_dispatched_val += dispatched
        else:
            self.total_requests_dispatched_train += dispatched
        tasks = {}
        for i, input_data in enumerate(vllm_inputs):
            prompt = TokensPrompt(**input_data)
            max_tokens = self.max_model_len - len(input_data["prompt_token_ids"])
            sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_kwargs)
            request_id = f"{batch_id}_{i}"
            tasks[request_id] = asyncio.create_task(
                # NOTE: use i:i+1 to avoid squeezing dimension
                self._run_request(prompt, sampling_params, request_id, gen_batch[i : i + 1], out_batch[i : i + 1], is_validation))
        await asyncio.gather(*tasks.values())
        completed = dispatched
        if is_validation:
            self.total_requests_completed_val += completed
        else:
            self.total_requests_completed_train += completed

