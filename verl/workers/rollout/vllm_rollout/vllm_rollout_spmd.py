# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            # NOTE: import os removed by Reasoning360. Definitely a bug of the official code.
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,   # NOTE: modified by Reasoning360. Originally config.get("seed", 0)
            **lora_kwargs,
            **engine_kwargs,
        )
        # NOTE: added by Reasoning360
        # self._monkey_patch_vllm_engine_v0()

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # NOTE: added by Reasoning360. timer for precise logging
    @staticmethod
    @contextmanager
    def timer():
        import time

        start = end = time.perf_counter()
        yield lambda: end - start
        end = time.perf_counter()

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id=lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}",lora_int_id=lora_int_id,lora_path="/simon-stub-path")] * batch_size

        # NOTE: added by Reasoning360
        if "num_samples" in prompts.meta_info:
            kwargs["n"] = prompts.meta_info["num_samples"]

        # Check for individual response lengths in non_tensor_batch
        individual_sampling_params = None
        per_prompt_max_length = None
        if "per_prompt_max_length" in prompts.non_tensor_batch:
            response_length = prompts.non_tensor_batch["per_prompt_max_length"]
            
            # Convert to list if it's a numpy array
            if isinstance(response_length, np.ndarray):
                response_length = response_length.tolist()
            
            # Ensure we have the right number for this worker's batch
            if len(response_length) != batch_size:
                # The framework distributes data across workers, so we may get a subset
                # Take the first batch_size elements (assumes ordered distribution)
                response_length = response_length[:batch_size] if len(response_length) > batch_size else response_length
                print(f"vLLM rollout (SPMD): Using {len(response_length)} response lengths for batch_size {batch_size}")
            
            # Set flag to use individual context manager approach
            individual_sampling_params = True
            per_prompt_max_length = response_length
            print(f"vLLM rollout (SPMD): Using individual response lengths (range: {min(response_length)}-{max(response_length)})")
        
        # Check for single response_length override in meta_info (backward compatibility)
        elif "response_length" in prompts.meta_info:
            response_length = prompts.meta_info["response_length"]
            # meta_info should only contain single values, not lists
            kwargs["max_tokens"] = response_length
            print(f"vLLM rollout (SPMD): Overriding max_tokens to {response_length}")
        
        print(f"kwargs by 360 (SPMD): {kwargs}")

        # users can customize different sampling_params at different run
        if individual_sampling_params is not None:
            # Use context manager approach to create individual sampling params while maintaining batch efficiency
            individual_sampling_params_list = []
            
            # Create individual sampling params using context manager approach (no rollback needed since we're not calling generate)
            for length in response_length:
                # Create kwargs for this specific prompt
                individual_kwargs = kwargs.copy()
                individual_kwargs["max_tokens"] = int(length)
                
                # Temporarily create a modified sampling params using the same logic as update_sampling_params
                # Save current state
                old_sampling_params_args = {}
                for key, value in individual_kwargs.items():
                    if hasattr(self.sampling_params, key):
                        old_value = getattr(self.sampling_params, key)
                        old_sampling_params_args[key] = old_value
                        setattr(self.sampling_params, key, value)
                
                # Create a copy with current state (same as what update_sampling_params would use)
                import copy
                individual_sampling_param = copy.deepcopy(self.sampling_params)
                individual_sampling_params_list.append(individual_sampling_param)
                
                # Restore original state
                for key, value in old_sampling_params_args.items():
                    setattr(self.sampling_params, key, value)
            
            # Now do batch inference with all prompts and individual sampling params
            with self.timer() as t:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=individual_sampling_params_list,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
            print(f"individual_sampling_params: Using context manager approach with batch inference")
            print(f"vLLM rollout (SPMD): Created {len(individual_sampling_params_list)} individual sampling params")
            print(f"vLLM rollout (SPMD): Sample individual sampling params: n={individual_sampling_params_list[0].n}, max_tokens={individual_sampling_params_list[0].max_tokens}")
        else:
            # Use single sampling params for all prompts (original behavior)
            with self.update_sampling_params(**kwargs), self.timer() as t:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        rollout_log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                response.append(response_ids)
                curr_log_prob = []
                for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                    curr_log_prob.append(logprob[response_ids[i]].logprob)
                rollout_log_probs.append(curr_log_prob)

        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)

        # print(f"vLLM rollout (SPMD): After generation - idx.shape={idx.shape}, response.shape={response.shape}")
        # print(f"vLLM rollout (SPMD): batch_size={batch_size}, len(outputs)={len(outputs)}")
        # print(f"vLLM rollout (SPMD): total responses collected={len(response)}")
        if individual_sampling_params is not None:
            print(f"vLLM rollout (SPMD): individual_sampling_params n values: {[sp.n for sp in individual_sampling_params_list]}")

        # Handle n > 1 case (multiple samples per prompt)
        if individual_sampling_params is not None:
            # When using individual sampling params, check if response tensor has more rows than idx
            # This happens when the sampling params have n > 1
            if response.shape[0] > idx.shape[0]:
                expansion_factor = response.shape[0] // idx.shape[0]
                print(f"vLLM rollout (SPMD): Detected expansion factor={expansion_factor} for individual_sampling_params")
                if response.shape[0] % idx.shape[0] == 0:  # Ensure it's a clean multiple
                    idx = _repeat_interleave(idx, expansion_factor)
                    attention_mask = _repeat_interleave(attention_mask, expansion_factor)
                    position_ids = _repeat_interleave(position_ids, expansion_factor)
                    batch_size = batch_size * expansion_factor
                    print(f"vLLM rollout (SPMD): Expanded batch_size from {batch_size // expansion_factor} to {batch_size}")
                    if "tools_kwargs" in non_tensor_batch.keys():
                        non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], expansion_factor)
                else:
                    raise ValueError(f"Response shape {response.shape[0]} is not a clean multiple of idx shape {idx.shape[0]}")
            effective_n = 1  # We've already handled expansion above
        else:
            # Use the n value from kwargs instead of self.sampling_params 
            # because self.sampling_params has reverted after context manager exit
            effective_n = kwargs.get('n', 1)
            
        print(f"vLLM rollout (SPMD): effective_n={effective_n}, do_sample={do_sample}")
            
        if effective_n > 1 and do_sample:
            print(f"vLLM rollout (SPMD): Applying effective_n={effective_n} expansion")
            idx = _repeat_interleave(idx, effective_n)
            attention_mask = _repeat_interleave(attention_mask, effective_n)
            position_ids = _repeat_interleave(position_ids, effective_n)
            batch_size = batch_size * effective_n
            print(f"vLLM rollout (SPMD): Expanded batch_size to {batch_size}")
            # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
            if "tools_kwargs" in non_tensor_batch.keys():
                non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], effective_n)

        # print(f"vLLM rollout (SPMD): After expansion - idx.shape={idx.shape}, response.shape={response.shape}")
        # print(f"vLLM rollout (SPMD): About to concatenate tensors")
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # NOTE: added by Reasoning360. temporarily disabled to avoid messy logging
        # tokens_per_second = torch.sum(response_attention_mask).item() / t()
        # print(
        #     f'Tokens per second: {tokens_per_second} t/s on device {os.environ["CUDA_VISIBLE_DEVICES"]} on host {os.uname().nodename}',
        #     flush=True,
        # )

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Prepare meta_info for the returned DataProto
        meta_info = prompts.meta_info.copy()
        
        # Expand per_prompt_max_length to match final batch size if needed
        if "per_prompt_max_length" in non_tensor_batch and per_prompt_max_length is not None:
            original_worker_lengths = len(per_prompt_max_length)
            final_batch_size = batch_size  # This is the final batch size after all expansions
            
            if final_batch_size > original_worker_lengths:
                # Calculate expansion factor (due to n > 1)
                expansion_factor = final_batch_size // original_worker_lengths
                if final_batch_size % original_worker_lengths != 0:
                    raise ValueError(f"Final batch size {final_batch_size} is not a clean multiple of original worker lengths {original_worker_lengths}")
                
                # Expand individual response lengths to match the final batch size
                expanded_lengths = []
                for length in per_prompt_max_length:
                    expanded_lengths.extend([length] * expansion_factor)
                
                # Update per_prompt_max_length with expanded lengths for metrics calculation
                non_tensor_batch["per_prompt_max_length"] = np.array(expanded_lengths, dtype=object)
                print(f"vLLM rollout (SPMD): Expanded per_prompt_max_length from {original_worker_lengths} to {len(expanded_lengths)} (factor: {expansion_factor})")
            # If no expansion needed, per_prompt_max_length is already in non_tensor_batch from earlier
        else:
            # Set target_max_response_length for backward compatibility (only if we didn't use individual response lengths)
            meta_info["target_max_response_length"] = self.config.response_length
            print(f"vLLM rollout (SPMD): Using target_max_response_length = {self.config.response_length}")

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
