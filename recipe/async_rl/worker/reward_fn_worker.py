import numpy as np
from transformers import AutoTokenizer

from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager
from verl.single_controller.base.worker import Worker
import ray


class AsyncRewardManagerActor(Worker):
    def __init__(self, config):
        Worker.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.actor_rollout_ref.model.path,
        )
        # train mode by default
        self.reward_fn = load_reward_manager(
            config=config,
            tokenizer=self.tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        self.dst_data_buffer = None
        self.val_dst_data_buffer = None
        # Aggregated counters
        self.total_rewards_computed: int = 0
        self.total_rewards_computed_train: int = 0
        self.total_rewards_computed_val: int = 0

    def validate(self):
        """Switch to validation mode."""
        self.reward_fn = load_reward_manager(
            config=self.config,
            tokenizer=self.tokenizer,
            num_examine=1,
            **self.config.reward_model.get("reward_kwargs", {}),
        )

    def train(self):
        """Switch to training mode."""
        self.reward_fn = load_reward_manager(
            config=self.config,
            tokenizer=self.tokenizer,
            num_examine=0,
            **self.config.reward_model.get("reward_kwargs", {}),
        )

    async def compute_rewards(self, data: DataProto, is_validation: bool = False):
        """Compute request and send to the next stage."""
        try:
            reward_result = self.reward_fn(data, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        except Exception as e:
            print(f"Error in reward_fn: {e}", flush=True)
            reward_tensor = self.reward_fn(data)
            reward_extra_infos_dict = {}
        data.batch["token_level_scores"] = reward_tensor
        if reward_extra_infos_dict:
            data.non_tensor_batch.update(
                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
            )
        # Update counters
        batch_len = len(data)
        self.total_rewards_computed += batch_len
        if is_validation:
            self.total_rewards_computed_val += batch_len
        else:
            self.total_rewards_computed_train += batch_len

        if is_validation:
            if self.val_dst_data_buffer is None:
                raise RuntimeError("Validation buffer not linked. Call link_validation_buffer() first.")
            await self.val_dst_data_buffer.add_data.remote(data)
        else:
            await self.dst_data_buffer.add_data.remote(data)

    async def link_buffer(self, dst_data_buffer):
        """Link the data buffer to send outputs to the next stage."""
        self.dst_data_buffer = dst_data_buffer

    async def link_validation_buffer(self, val_dst_data_buffer):
        self.val_dst_data_buffer = val_dst_data_buffer

    async def get_counters(self):
        return {
            "total": self.total_rewards_computed,
            "train": self.total_rewards_computed_train,
            "val": self.total_rewards_computed_val,
        }
