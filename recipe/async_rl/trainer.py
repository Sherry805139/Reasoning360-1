import asyncio

from omegaconf import DictConfig
import ray
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, AdvantageEstimator

from async_rl.controller.train import TrainController, PreprocessController
from async_rl.controller.rollout import RolloutController
from async_rl.controller.reward_fn import RewardFnController
from async_rl.data_buffer import DataBuffer, DataLoaderBuffer, RequestGatherBuffer


class Trainer:
    """
    A light trainer launching all controllers. Each controller manages a stage of the
    training pipeline. The controller pulls data from the input data buffer, dispatching
    them to the workers in this stage. The workers in this stage will send data to the
    next stage.
    """

    # TODO: use resource pool manager instead of a rough config.
    def __init__(self, config: DictConfig, resource_pool_specs, processor=None):
        self.config = config
        # TODO: add a config to enable distributed controller.
        # This avoids CPU network congestion, but is less friendly for debugging.
        self.train_controller = TrainController(config)
        self.reward_fn_controller = RewardFnController(config)
        self.rollout_controller = RolloutController(config)
        self.train_preprocess_controller = PreprocessController(config)
        self.src_data_buffer = None
        self.val_src_data_buffer = None
        self.rollout_out_buffer = None
        self.reward_out_buffer = None
        self.train_process_out_buffer = None
        self._init_worker_group(resource_pool_specs, processor)

    def _arg_check(self, config: DictConfig):
        if config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise ValueError("REMAX is not supported yet.")

    def _init_worker_group(self, resource_pool_specs, processor):
        ## 1. Init worker groups for each stage.
        self._check_resource_available(resource_pool_specs)
        self.train_controller.init_wg(resource_pool_specs["train"])
        self.rollout_controller.init_wg(resource_pool_specs["rollout"])
        self.train_preprocess_controller.init_wg(resource_pool_specs["preprocess"])
        self.reward_fn_controller.init_wg(resource_pool_specs["reward"])

        ## 2. Init data buffers.
        # TODO: ray options to collocate them with the next stage controller?
        # NOTE: src_data_buffer is specialized to load data.
        dataloader_batch_size = 1
        # This is the training data buffer.
        self.src_data_buffer = ray.remote(num_cpus=1)(DataLoaderBuffer).remote(
            self.config.data,
            self.config.actor_rollout_ref.model.path,
            processor,
            dataloader_batch_size,
        )
        # This is the validation data buffer.
        self.val_src_data_buffer = ray.remote(num_cpus=1)(DataLoaderBuffer).remote(
            self.config.data,
            self.config.actor_rollout_ref.model.path,
            processor,
            dataloader_batch_size,
            max_samples=self.config.data.val_max_samples,  # use validation max samples
            is_validation=True,
        )
        # This is the training rollout output buffer.
        self.rollout_out_buffer = ray.remote(num_cpus=1)(DataBuffer).remote()
        # This is the validation rollout output buffer.
        self.val_rollout_out_buffer = ray.remote(num_cpus=1)(DataBuffer).remote()
        # NOTE: this data buffer is specialized to make sure rollouts from
        # the same requests are sent together (GRPO needs this to compute advantages).
        self.reward_out_buffer = ray.remote(num_cpus=1)(RequestGatherBuffer).remote(
            num_repeats=self.config.actor_rollout_ref.rollout.n
        )
        # This is the validation reward output buffer.
        self.val_reward_out_buffer = ray.remote(num_cpus=1)(RequestGatherBuffer).remote(
            num_repeats=self.config.actor_rollout_ref.rollout.n
        )
        self.train_process_out_buffer = ray.remote(num_cpus=1)(DataBuffer).remote()
        ## 3. Link buffers.
        self.rollout_controller.link_buffer(
            self.src_data_buffer, self.rollout_out_buffer
        )
        self.rollout_controller.link_validation_buffer(
            self.val_src_data_buffer, self.val_rollout_out_buffer
        )
        self.rollout_controller.link_servers_to_buffer()
        self.rollout_controller.link_servers_to_validation_buffer()

        self.reward_fn_controller.link_buffer(
            self.rollout_out_buffer, self.reward_out_buffer
        )
        self.reward_fn_controller.link_validation_buffer(
            self.val_rollout_out_buffer, self.val_reward_out_buffer
        )
        self.train_preprocess_controller.link_buffer(
            self.reward_out_buffer, self.train_process_out_buffer
        )
        # TODO: a buffer to receive metrics.
        self.train_controller.link_buffer(self.train_process_out_buffer, None)
        ## 4. set dataset size.
        dataset_size = ray.get(self.src_data_buffer.get_dataset_size.remote())
        num_repeats = self.config.actor_rollout_ref.rollout.n
        if dataset_size > 0:
            self.rollout_controller.set_dataset_size(dataset_size)
            self.reward_fn_controller.set_dataset_size(dataset_size * num_repeats)
            self.train_preprocess_controller.set_dataset_size(dataset_size * num_repeats)
            self.train_controller.set_dataset_size(dataset_size)

        ## 5. Link rollout controller and validation components.
        self.train_controller.link_rollout_controller(self.rollout_controller)
        self.train_controller.link_validation_components(self.reward_fn_controller, self.val_rollout_out_buffer, self.val_reward_out_buffer)

    async def _reset_data_loader_buffer(self):
        await self.src_data_buffer.reset_counter.remote()
        await self.val_src_data_buffer.reset_counter.remote()

    async def fit(self):
        # TODO: load checkpoint, if any. reference: verl ray_trainer.py, _load_checkpoint
        # TODO: support validation.
        # TODO: if do_profile, add profile metrics.

        # Reset data loader buffer before starting training
        await self._reset_data_loader_buffer()

        rollout_task = asyncio.create_task(self.rollout_controller.rollout_loop())
        reward_task = asyncio.create_task(self.reward_fn_controller.reward_loop())
        preprocess_task = asyncio.create_task(self.train_preprocess_controller.preprocess_loop())
        train_task = asyncio.create_task(self.train_controller.train_loop())

        await asyncio.gather(rollout_task, reward_task, preprocess_task, train_task)

    def _check_resource_available(self, resource_pool_spec):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")