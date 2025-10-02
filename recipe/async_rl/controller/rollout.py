import asyncio
import uuid

import numpy as np
from omegaconf import DictConfig
import ray

from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)

from async_rl.async_rollout_server import InterruptableAsyncvLLMServer
from async_rl.controller.base_controller import StageController
from async_rl.worker.rollout_worker import AsyncRolloutWorker


class RolloutController(StageController):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.reward_fn_wg: RayWorkerGroup = None
        self.servers: dict[int, InterruptableAsyncvLLMServer] = None
        self.num_requests_trained = 0
        self.num_requests_sent = 0
        self.dst_data_buffer = None
        self.val_dst_data_buffer = None
        self.max_staleness = self.config.trainer.get("max_staleness", 8)

    def init_wg(self, world_desc: list[int]):
        """Init the rollout controller worker group."""
        rollout_pool = RayResourcePool(
            process_on_nodes=world_desc,
            name_prefix="reward_fn_pool",
            max_colocate_count=1,
        )

        rollout_worker_cls = RayClassWithInitArgs(
            ray.remote(AsyncRolloutWorker),
            config=self.config.actor_rollout_ref,
            role="rollout",
        )

        self.rollout_wg = RayWorkerGroup(
            resource_pool=rollout_pool,
            ray_cls_with_init=rollout_worker_cls,
            name_prefix="rollout",
        )
        self.rollout_wg.init_model()

        rollout_tp_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        )
        rollout_dp_size = self.rollout_wg.world_size // rollout_tp_size

        # Get worker node information using the modern verl approach
        workers_info = ray.get(
            [
                worker.__ray_call__.remote(
                    lambda self: ray.get_runtime_context().get_node_id()
                )
                for worker in self.rollout_wg._workers
            ]
        )
        server_cls = ray.remote(num_cpus=1)(InterruptableAsyncvLLMServer)
        servers = {
            rollout_dp_rank: server_cls.options(
                # make sure AsyncvLLMServer colocates with its corresponding workers
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=workers_info[rollout_dp_rank * rollout_tp_size],
                    soft=False,
                ),
                name=f"async_llm_server_{rollout_dp_rank}",
                runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
            ).remote(
                self.config,
                rollout_dp_size,
                rollout_dp_rank,
                self.rollout_wg.name_prefix,
            )
            for rollout_dp_rank in range(rollout_dp_size)
        }

        ray.get([server.init_engine.remote() for server in servers.values()])
        self.servers = servers

    def link_servers_to_buffer(self):
        """Link servers to the destination data buffer after buffers are linked."""
        if self.dst_data_buffer is not None:
            # Note: This is called during initialization, so it's okay to use ray.get here
            # as it's not in an async context
            ray.get(
                [
                    server.link_data_buffer.remote(self.dst_data_buffer)
                    for server in self.servers.values()
                ]
            )

    def link_servers_to_validation_buffer(self):
        if self.val_dst_data_buffer is not None:
            ray.get(
                [
                    server.link_validation_buffer.remote(self.val_dst_data_buffer)
                    for server in self.servers.values()
                ]
            )

    #### Data buffer size related functions.
    def dst_drain_size(self, src_drain_size: int):
        """Multiply the src drain size by the sampling param n."""
        return src_drain_size * self.config.actor_rollout_ref.rollout.n

    #### Main loop
    def _continue_adding_requests(self):
        return (self.num_requests_sent - self.num_requests_trained) < (
            self.max_staleness * self.config.data.train_batch_size
        )

    async def rollout_loop(self, is_validation: bool = False):
        self.num_requests_trained = 0
        self.num_requests_sent = 0
        if is_validation:
            # For validation, only run once (not for multiple epochs)
            cur_server = -1
            gen_batch_size = 1
            async for i, rollout_batch in self.validation_iterator(gen_batch_size):
                cur_server = (cur_server + 1) % len(self.servers)
                rollout_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(rollout_batch.batch))],
                    dtype=object,
                )
                rollout_batch = rollout_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                _ = self.servers[cur_server].generate_and_send.remote(
                    rollout_batch, f"val_0_{i}", is_validation=True
                )
                self.num_requests_sent += gen_batch_size
        else:
            # For training, run for multiple epochs
            for epoch in range(self.config.trainer.total_epochs):
                cur_server = -1
                # TODO: add load balancing here.
                # NOTE: When the controller is terminated, the get_data_iter will directly return,
                # making this for loop exit.
                # TODO: make this gen batch size configurable (do not use data.gen_batch_size).
                gen_batch_size = 1
                async for i, rollout_batch in self.iterator(gen_batch_size):
                    cur_server = (cur_server + 1) % len(self.servers)
                    # TODO: trace global_steps
                    # rollout_batch.meta_info["global_steps"] = self.global_steps
                    rollout_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(rollout_batch.batch))],
                        dtype=object,
                    )
                    rollout_batch = rollout_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    _ = self.servers[cur_server].generate_and_send.remote(
                        rollout_batch, f"train_{epoch}_{i}"
                    )
                    self.num_requests_sent += gen_batch_size
                    while not self._continue_adding_requests():
                        await asyncio.sleep(0)

    async def get_counters(self):
        counters = {}
        for server_id, server in self.servers.items():
            server_counters = await server.get_counters.remote()
            counters[f"server_{server_id}"] = server_counters
        return counters

    async def update_weights(self, checkpoint_path: str):
        tasks = []
        for i in self.servers:
            tasks.append(self.servers[i].update_weights.remote(checkpoint_path))
        await asyncio.gather(*tasks)
        # NOTE: the train_batch_size is the number of requests before sampling
        self.num_requests_trained += self.config.data.train_batch_size
