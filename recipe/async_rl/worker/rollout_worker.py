from omegaconf import DictConfig, OmegaConf
import torch
import os
import datetime
import logging
from typing import Any

from transformers import AutoConfig
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_nccl_backend, get_torch_device, get_device_name
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.profiler import (
    DistProfiler,
    DistProfilerExtension,
    GPUMemoryLogger,
    log_gpu_memory_usage,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as FsdpAsyncActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class FsdpAsyncRolloutWorker(FsdpAsyncActorRolloutRefWorker):
    def __init__(self, config, role: str):
        assert role == "rollout"
        super().__init__(config, role)


class _PlaceHolderShardingManager:
    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class AsyncRolloutWorker(Worker, DistProfilerExtension):
    """
    NOTE: use our own rollout worker to avoid dependency on actors in sharding manager.
    """
    def __init__(self, config: DictConfig, role: str):
        assert role == "rollout"
        Worker.__init__(self)
        self.config = config

        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

        set_random_seed(seed=self.config.actor.fsdp_config.seed)   # TODO: a config for rollout's seed?

        self.role = role
        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

    # Main api exposed to worker groups
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # init tokenizer and self.hf_config
        model_path = self.config.model.path
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        self._init_hf_config(
            model_path=model_path,
            tokenizer_or_path=model_path,
            override_model_config=override_model_config,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )

        self.generation_config = get_generation_config(self.local_path)

        # TODO: add a config.
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        rollout = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        self.rollout = rollout
        self.sharding_manager = _PlaceHolderShardingManager()
        self.rollout.sharding_manager = self.sharding_manager

        # NOTE: we don't need sharding manager yet. Will attach training
        # with inference later, potentially by the sharding manager.

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        raise RuntimeError("In async_rl, generate_sequences is not explicitly called by worker groups.")

    # ============================ init related ============================
    def _build_rollout(self, trust_remote_code=False):

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        if self.config.rollout.name != "vllm":
            raise NotImplementedError(f"Rollout name {self.config.rollout.name} is not supported")

        infer_tp = self.config.rollout.tensor_model_parallel_size

        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        log_gpu_memory_usage("Before building vllm rollout", logger=None)

        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        assert self.config.rollout.mode == "async", "Only async mode is supported for rollout"

        # Create HFModelConfig for the new vLLMAsyncRollout API
        from verl.workers.config import HFModelConfig
        
        model_config = HFModelConfig(
            path=self.config.model.path,
            trust_remote_code=trust_remote_code,
            use_shm=self.config.model.get("use_shm", False),
        )
        
        vllm_rollout_cls = vLLMAsyncRollout
        rollout = vllm_rollout_cls(
            config=self.config.rollout,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage("After building vllm rollout", logger=logger)

        # perform weight resharding between actor and rollout
        # from verl.models.mcore import get_mcore_weight_converter
        # weight_converter = get_mcore_weight_converter(self.hf_config, self.dtype)

        self.vllm_tp_size = infer_tp
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size
        return rollout

    def _init_hf_config(self, model_path, tokenizer_or_path, override_model_config, trust_remote_code=False):
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        self.hf_config = hf_config

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: str | bytes, *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(
                f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] execute_method: "
                f"{method if isinstance(method, str) else 'Callable'}"
            )
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.sleep()
        # return something to block the caller
        return True
