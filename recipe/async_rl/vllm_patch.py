import asyncio
from contextlib import nullcontext
import logging
from multiprocessing.connection import Connection
import os
import psutil
import signal
import threading
from typing import Optional
import weakref

import torch
import zmq
import zmq.asyncio

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor.model_loader.loader import (
    DefaultModelLoader, get_model_loader, _process_weights_after_loading
)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (get_exception_traceback, get_open_zmq_ipc_path,
                        kill_process_tree, make_zmq_socket)
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler, SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import AnyFuture, AsyncMPClient, BackgroundResources
from vllm.v1.engine.async_llm import AsyncLLM as V1AsyncLLM
from vllm.v1.engine.processor import Processor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (LoggingStatLogger, PrometheusStatLogger,
                                     StatLoggerBase)
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import BackgroundProcHandle
from vllm.v1.worker.gpu_worker import Worker as V1Worker

logger = init_logger(__name__)

###### Worker side: update weights
# NOTE: for reset at the worker level, we simply pass a special scheduler output
# to clean up everything.
# TODO: use worker_extension_cls to inject the new functions.

class MutableWeightRayWorker(V1Worker):
    def release_weights(self):
        """
        The previous verision of weight must be released.
        NOTE: not tested because there seems no point to enable sleep mode.
        """
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            for ptr, data in allocator.pointer_to_data.items():
                if data.tag == "weights":
                    unmap_and_release(data.handle)

    def runner_update_model(self, ckpt_path: str):
        """
        Unfolding self.model_runner.load_model() by skipping initialization.
        """
        # copied from the default model loader but skip initialization
        device_config = self.vllm_config.device_config
        model_config = self.vllm_config.model_config
        loader = get_model_loader(self.vllm_config.load_config)
        if not isinstance(loader, DefaultModelLoader):
            raise NotImplementedError("Only DefaultModelLoader is supported yet")

        # monkey patch the model path to the new checkpoint's
        backup_path = model_config.model
        model_config.model = ckpt_path
        ##

        target_device = torch.device(device_config.device)
        model = self.model_runner.model
        with set_default_torch_dtype(model_config.dtype):

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                loader._get_all_weights(model_config, model))
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")

            _process_weights_after_loading(model, model_config, target_device)

        if self.model_runner.lora_config:
            raise NotImplementedError("Lora is not supported")

        model_config.model = backup_path

    def update_weights(self, ckpt_path: str):
        self.release_weights()

        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            context = nullcontext()
        with context:
            self.runner_update_model(ckpt_path)


###### Server side: preempt and launch worker update_weights
class InterruptableScheduler(V1Scheduler):
    def reset(self):
        """
        This call will reset the scheduler so that all requests are back to the prompt phrase.
        NOTE: it does not change anything on the workers. The KV block id is reallocated when requests are re-scheduled.
        """
        running_requests = list(self.running)
        self.running.clear()
        for i, req in enumerate(running_requests):
            req_id = req.request_id
            num_computed_tokens = req.num_computed_tokens
            # print(f"[SCHEDULER_RESET] Request {i+1}/{len(running_requests)}: {req_id} tokens={num_computed_tokens} status={req.status}",flush=True)
            # Free KV cache blocks
            self.kv_cache_manager.free(req)
            # Update request status
            req.status = RequestStatus.PREEMPTED
            req.num_computed_tokens = 0
        self.waiting.extendleft(running_requests)
        self.reset_prefix_cache()
        return len(running_requests)

    def preempt_all_step(self) -> SchedulerOutput:
        self.finished_req_ids = set()
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=[],
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids={},
            grammar_bitmask=None,
        )


class InterruptableEngineCoreProc(EngineCoreProc):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        ready_pipe: Connection,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        vllm_config.scheduler_config.scheduler_cls = InterruptableScheduler
        assert vllm_config.parallel_config.worker_cls == "async_rl.vllm_patch.MutableWeightRayWorker"
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            ready_pipe=ready_pipe,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        self.scheduler: InterruptableScheduler

    def update_weights(self, ckpt_path: str):
        """
        This function preempts all requests and update the weights.
        NOTE: Disaggregation is not considered yet.
        """
        num_preempted = self.scheduler.reset()
        preempt_all_scheduler_output = self.scheduler.preempt_all_step()
        self.model_executor.execute_model(preempt_all_scheduler_output)
        self.model_executor.collective_rpc("update_weights", args=(ckpt_path,))
        return num_preempted

    @staticmethod
    def run_engine_core(*args, **kwargs):
        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()
        engine_core = None
        try:
            engine_core = InterruptableEngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGUSR1)

        finally:
            if engine_core is not None:
                engine_core.shutdown()


###### Client side: passing requests to the server.
class InterruptableEngineCoreClient(AsyncMPClient):
    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        asyncio_mode = True
        def sigusr1_handler(signum, frame):
            logger.fatal("Got fatal signal from worker processes, shutting "
                         "down. See stack trace above for root cause issue.")
            kill_process_tree(os.getpid())

        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGUSR1, sigusr1_handler)
        else:
            logger.warning("SIGUSR1 handler not installed because we are not "
                           "running in the main thread. In this case the "
                           "forked engine process may not be killed when "
                           "an exception is raised, and you need to handle "
                           "the engine process shutdown manually.")

        # Serialization setup.
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        sync_ctx = zmq.Context()
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected,  even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)

        # Paths for IPC.
        self.output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()

        # Start EngineCore in background process.
        self.resources.proc_handle = BackgroundProcHandle(
            input_path=input_path,
            output_path=self.output_path,
            process_name="EngineCore",
            target_fn=InterruptableEngineCoreProc.run_engine_core,
            process_kwargs={
                "vllm_config": vllm_config,
                "executor_class": executor_class,
                "log_stats": log_stats,
            })

        # Create input socket.
        self.resources.input_socket = make_zmq_socket(self.ctx, input_path,
                                                      zmq.constants.PUSH)
        self.input_socket = self.resources.input_socket
        self.utility_results: dict[int, AnyFuture] = {}

        self.outputs_queue: Optional[asyncio.Queue[EngineCoreOutputs]] = None
        self.queue_task: Optional[asyncio.Task] = None

    async def update_weights(self, ckpt_path: str):
        return await self._call_utility_async("update_weights", ckpt_path)


class InterruptableAsyncLLM(V1AsyncLLM):
    """
    NOTE: the only delta is the engine_core uses InterruptableEngineCoreProc,
    and `update_weights` is added.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        assert start_engine_loop

        self.model_config = vllm_config.model_config

        self.log_requests = log_requests
        self.log_stats = log_stats
        self.stat_loggers: list[StatLoggerBase] = []
        if self.log_stats:
            if logger.isEnabledFor(logging.INFO):
                self.stat_loggers.append(LoggingStatLogger())
            self.stat_loggers.append(PrometheusStatLogger(vllm_config))

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)
        self.tokenizer.ping()

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            input_registry=input_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)

        # EngineCore (starts the engine in background process).
        self.engine_core = InterruptableEngineCoreClient(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.output_handler: Optional[asyncio.Task] = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "InterruptableAsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # FIXME(rob): refactor VllmConfig to include the StatLoggers
        # include StatLogger in the Oracle decision.
        if stat_loggers is not None:
            raise ValueError("Custom StatLoggers are not yet supported on V1. "
                             "Explicitly set VLLM_USE_V1=0 to disable V1.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
        )

    async def update_weights(self, ckpt_path: str):
        return await self.engine_core.update_weights(ckpt_path)
