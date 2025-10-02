from omegaconf import DictConfig, OmegaConf
import ray
import os
from typing import Dict, Any
from tqdm import tqdm
import asyncio

from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.tracking import Tracking
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import compute_data_metrics

from async_rl.controller.base_controller import StageController
from async_rl.controller.rollout import RolloutController
from async_rl.worker.train_preprocess import AsyncPreprocessWorker
from async_rl.worker.train_worker import AsyncActorWorker
from async_rl.utils import compute_validation_metrics, aggregate_validation_metrics


class TrainController(StageController):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.actor_wg: RayWorkerGroup = None
        self.rollout_controller: RolloutController = None

    def init_wg(self, world_desc: list[int]):
        actor_pool = RayResourcePool(
            process_on_nodes=world_desc, name_prefix="actor_pool", max_colocate_count=1
        )
        actor_worker_cls = RayClassWithInitArgs(
            ray.remote(AsyncActorWorker),
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.actor_wg = RayWorkerGroup(
            resource_pool=actor_pool,
            ray_cls_with_init=actor_worker_cls,
            name_prefix="actor",
        )
        self.actor_wg.init_model()

    def link_buffer(self, src_data_buffer, dst_data_buffer):
        assert dst_data_buffer is None, "currently not supported."
        super().link_buffer(src_data_buffer, None)

    def link_rollout_controller(self, rollout_controller: RolloutController):
        self.rollout_controller = rollout_controller

    def link_validation_components(self, reward_fn_controller, val_rollout_out_buffer, val_reward_out_buffer):
        """Link the reward function controller and validation buffer for validation steps"""
        self.reward_fn_controller = reward_fn_controller
        self.val_reward_out_buffer = val_reward_out_buffer
        self.val_rollout_out_buffer = val_rollout_out_buffer

    def _collect_and_log_metrics(self, train_batch, training_result, global_step: int) -> Dict[str, Any]:
        training_metrics = {}
        if hasattr(training_result, "meta_info") and training_result.meta_info and "metrics" in training_result.meta_info:
            training_metrics = training_result.meta_info["metrics"]

        # Compute data-specific metrics using verl's compute_data_metrics
        try:
            # in grpo, we don't use the critic.
            data_metrics = compute_data_metrics(train_batch, use_critic=False)
        except Exception as e:
            print(f"Warning: Failed to compute data metrics at step {global_step}: {e}")
            data_metrics = {}

        # Combine all metrics
        combined_metrics = {
            "global_step": global_step,
            **training_metrics,
            **data_metrics,
        }

        if global_step % 10 == 0:
            print(f"\nStep {global_step} Metrics:")
            for key, value in combined_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

        return combined_metrics

    def _save_checkpoint(self, global_step: int):
        # path: given_path + `/global_step_{global_step}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{global_step}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{global_step}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            global_step,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        # TODO: save dataloader
        # TODO: save latest checkpointed iteration tracker

        return actor_local_path

    async def _validation_step(self):
        validation_results = []
        
        # Reset ALL validation buffers before starting validation
        await self.rollout_controller.val_src_data_buffer.reset_counter.remote()
        await self.val_rollout_out_buffer.reset_counter.remote()
        await self.val_reward_out_buffer.reset_counter.remote()
        print("DEBUG: Reset all validation buffers")

        val_dataset_size = await self.rollout_controller.val_src_data_buffer.get_dataset_size.remote()

        # Start both rollout and reward processing loops concurrently
        # The reward function controller will automatically process validation rollouts
        # as they become available in the val_rollout_out_buffer
        async def run_validation_rollouts():
            await self.rollout_controller.rollout_loop(is_validation=True)
            
        async def run_validation_rewards():
            await self.reward_fn_controller.reward_loop(is_validation=True)

        rollout_task = asyncio.create_task(run_validation_rollouts())
        reward_task = asyncio.create_task(run_validation_rewards())

        # Wait for rollout generation to complete first
        await rollout_task

        # Give reward computation a moment to catch up
        await asyncio.sleep(1.0)

        collected_samples = 0
        try:
            timeout_counter = 0
            max_timeout = 100
            val_buffer = self.val_reward_out_buffer
            while timeout_counter < max_timeout and collected_samples < val_dataset_size:
                try:
                    remaining_samples = val_dataset_size - collected_samples
                    # Only request data if we have remaining samples to collect
                    if remaining_samples > 0:
                        val_data, drain_signal = await val_buffer.get_data.remote(remaining_samples)
                    else:
                        break
                    if val_data is not None:
                        validation_metrics = compute_validation_metrics(val_data, self.config.actor_rollout_ref.model.path)
                        validation_results.append(validation_metrics)
                        collected_samples += len(val_data)
                        print(f"Collected {collected_samples}/{val_dataset_size} validation samples")
                        if drain_signal >= 0 or collected_samples >= val_dataset_size:
                            break
                    else:
                        await asyncio.sleep(0.2)
                        timeout_counter += 1
                except Exception as e:
                    print(f"Error collecting validation data: {e}")
                    await asyncio.sleep(0.1)
                    timeout_counter += 1

        except Exception as e:
            print(f"Error in validation collection: {e}")

        # Wait for reward task to complete
        if collected_samples >= val_dataset_size:
            await reward_task
        else:
            print(f"Remaining samples: {val_dataset_size - collected_samples}")
            print("Reward computation timed out, cancelling...")
            reward_task.cancel()

        # Aggregate validation results
        if validation_results:
            aggregated_metrics = aggregate_validation_metrics(validation_results)
            print(f"Validation completed: {aggregated_metrics}")
            return aggregated_metrics
        else:
            print("No validation results collected")
            return {"validation/samples": 0, "validation/avg_reward": 0.0}


    async def train_loop(self):
        self.global_steps = 0

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.total_training_steps = self.config.actor_rollout_ref.actor.total_training_steps
        # self.max_steps_duration = 0
        # TODO: get total training steps from the data loader buffer.
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        batch_size = self.config.data.train_batch_size

        for epoch in range(self.config.trainer.total_epochs):
            async for batch_idx, train_batch in self.iterator(batch_size):
                metrics = {}
                self.global_steps += 1
                is_last_step = self.global_steps >= self.total_training_steps

                # TODO: balance batch here.
                # verl here calls the dynamic batch balancing

                # Instrumentation before actor update
                # Buffer diagnostics before update
                rollout_diag = await self.rollout_controller.get_counters()
                reward_diag = await self.reward_fn_controller.dst_data_buffer.get_counters.remote()
                print(f"[TRAIN] Before update step={self.global_steps}: rollout_out={rollout_diag} reward_out={reward_diag}", flush=True)

                result = self.actor_wg.update_actor(train_batch)
                actor_output_metrics = reduce_metrics(result.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # Collect and log comprehensive metrics
                metrics.update(self._collect_and_log_metrics(train_batch, result, self.global_steps))
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})

                # NOTE: _save_checkpoint() will save the checkpoint to the local directory.
                # currently, we end up saving fsdp + hf weights, later we will save only the hf weights.
                checkpoint_path = self._save_checkpoint(global_step=0)
                # Pass the huggingface subdirectory path to vLLM, not the actor directory
                # self.rollout_controller.update_weights.remote(checkpoint_path)
                await self.rollout_controller.update_weights(
                    os.path.join(checkpoint_path, "huggingface")
                )

                # Instrumentation after weight update
                rollout_diag = await self.rollout_controller.get_counters()
                reward_diag = await self.reward_fn_controller.dst_data_buffer.get_counters.remote()
                print(f"[TRAIN] After update step={self.global_steps}: rollout_out={rollout_diag} reward_out={reward_diag}", flush=True)

                # TODO: if validation step, switch to the validation mode.
                # Check if validation should be performed
                is_valid_step = self.config.trainer.get("test_freq", -1) > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.test_freq == 0
                )

                if is_valid_step:
                    print(f"\nRunning validation at step {self.global_steps}...")
                    validation_metrics = await self._validation_step()
                    metrics.update(validation_metrics)
                
                # Log training and validation metrics
                logger.log(data=metrics, step=self.global_steps)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                # esi_close_to_expiration = should_save_ckpt_esi(
                #     max_steps_duration=self.max_steps_duration,
                #     redundant_time=self.config.trainer.esi_redundant_time,
                # )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    # or esi_close_to_expiration
                ):
                    # if esi_close_to_expiration:
                    #     print("Force saving checkpoint: ESI instance expiration approaching.")
                    # FIXME: save ckpt here (NOTE: also need to save ckpt of other components, if necessary).
                    self._save_checkpoint(global_step=self.global_steps)

                # TODO: update metrics. extract metrics from metadata?
                progress_bar.update(1)


class PreprocessController(StageController):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.preprocess_wg: RayWorkerGroup = None
        self.num_repeats = self.config.actor_rollout_ref.rollout.n

    def init_wg(self, world_desc: list[int]):
        preprocess_pool = RayResourcePool(
            process_on_nodes=world_desc,
            name_prefix="preprocess_pool",
            max_colocate_count=1,
            use_gpu=False,
        )

        preprocess_worker_cls = RayClassWithInitArgs(
            ray.remote(AsyncPreprocessWorker),
            config=self.config,
            role="preprocess",
        )
        self.preprocess_wg = RayWorkerGroup(
            resource_pool=preprocess_pool,
            ray_cls_with_init=preprocess_worker_cls,
            name_prefix="preprocess",
        )
        self.num_preprocess_workers = self.preprocess_wg.world_size

    def link_buffer(self, src_data_buffer, dst_data_buffer):
        super().link_buffer(src_data_buffer, dst_data_buffer)
        # self.preprocess_wg.link_buffer.remote(dst_data_buffer)
        # for worker in self.preprocess_wg.workers:
        # ray.get(worker.link_buffer.remote(dst_data_buffer))
        self.preprocess_wg.execute_all_async("link_buffer", dst_data_buffer)

    async def preprocess_loop(self):
        batch_size = self.num_preprocess_workers * self.num_repeats
        for epoch in range(self.config.trainer.total_epochs):
            async for i, preprocess_batch in self.iterator(batch_size):
                # ray.get(self.preprocess_wg.preprocess.remote(preprocess_batch))
                self.preprocess_wg.execute_all_sync("preprocess", preprocess_batch)
