from omegaconf import DictConfig
import ray

from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)

from async_rl.controller.base_controller import StageController
from async_rl.worker.reward_fn_worker import AsyncRewardManagerActor


class RewardFnController(StageController):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.reward_fn_wg: RayWorkerGroup = None
        self.num_reward_workers = 0

    def init_wg(self, world_desc: list[int]):
        reward_pool = RayResourcePool(
            process_on_nodes=world_desc,
            name_prefix="reward_pool",
            max_colocate_count=1,
            use_gpu=False,
        )

        reward_worker_cls = RayClassWithInitArgs(
            ray.remote(AsyncRewardManagerActor),
            config=self.config,
            # role="reward",
        )
        self.reward_fn_wg = RayWorkerGroup(
            resource_pool=reward_pool,
            ray_cls_with_init=reward_worker_cls,
            name_prefix="reward",
        )
        self.num_reward_workers = self.reward_fn_wg.world_size

    def link_buffer(self, src_data_buffer, dst_data_buffer):
        super().link_buffer(src_data_buffer, dst_data_buffer)
        self.reward_fn_wg.execute_all_async("link_buffer", dst_data_buffer)

    def link_validation_buffer(self, val_src_data_buffer, val_dst_data_buffer):
        super().link_validation_buffer(val_src_data_buffer, val_dst_data_buffer)
        self.reward_fn_wg.execute_all_async("link_validation_buffer", val_dst_data_buffer)

    async def reward_loop(self, is_validation: bool = False):
        batch_size = self.num_reward_workers
        if is_validation:
            # For validation, only run once (not for multiple epochs)
            async for _, reward_req in self.validation_iterator(batch_size):
                self.reward_fn_wg.execute_all_sync("compute_rewards", reward_req, is_validation)
        else:
            # For training, run for multiple epochs
            for _ in range(self.config.trainer.total_epochs):
                async for _, reward_req in self.iterator(batch_size):
                    self.reward_fn_wg.execute_all_sync("compute_rewards", reward_req)
