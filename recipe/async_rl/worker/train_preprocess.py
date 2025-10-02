from omegaconf import DictConfig
import ray

from verl.protocol import DataProto
from verl.single_controller.base.worker import Worker
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage


class AsyncPreprocessWorker(Worker):
    def __init__(self, config: DictConfig, role: str):
        assert role == "preprocess"
        Worker.__init__(self)
        self.config = config
        self.role = role

    async def link_buffer(self, dst_data_buffer):
        """Link the data buffer to send outputs to the next stage."""
        self.dst_data_buffer = dst_data_buffer

    async def preprocess(self, data: DataProto):
        # TODO: this is a hack.
        # data["old_log_probs"] = data["rollout_log_probs"]
        data.batch["old_log_probs"] = data.batch["rollout_log_probs"]
        # copied from ray_trainer.py: fit

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            # TODO: add metrics
        else:
            # data["token_level_rewards"] = data["token_level_scores"]
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]
        norm_adv_by_std_in_grpo = self.config.algorithm.get(
            "norm_adv_by_std_in_grpo", True
        )
        data = compute_advantage(
            data,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        await self.dst_data_buffer.add_data.remote(data)
        # TODO: dump generations
        # if rollout_data_dir := self.config.trainer.get("rollout_data_dir", None):
        #     inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        #     outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        #     scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        #     self._dump_generations(
        #         inputs=inputs,
        #         outputs=outputs,
        #         scores=scores,
        #         reward_extra_infos_dict=reward_extra_infos_dict,
        #         dump_path=rollout_data_dir,
        #     )
