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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd 
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          _timer, apply_kl_penalty, compute_advantage, compute_response_mask)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_throughout_metrics, compute_timing_metrics,
                                           reduce_metrics, compute_difficulty_histogram_metrics)


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the drop count attributes to ensure they're always available
        self.n_drop_easy = 0
        self.n_drop_hard = 0

    def _create_priority_dataloader(self, epoch_idx):
        """
        Create the dataloader every time before the epoch starts.
        """
        from torch.utils.data import SequentialSampler
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader
        
        if epoch_idx == 0:# or self.config.trainer.pass_rate_threshold == 1.0: 
            # Use whole dataset with random/sequential sampler

            
            # Strategy 1: Use random pass rates
            # np.random.seed(42)  # For reproducible debugging
            # # random_pass_rates = np.random.uniform(0, 1, len(self.train_dataset.dataframe))
            # random_pass_rates = np.random.uniform(0.7, 0.8, len(self.train_dataset.dataframe))
            # self.train_dataset.dataframe["on_policy_pass_rate"] = random_pass_rates
            # print(f"DEBUG: Generated {len(random_pass_rates)} random pass rates between 0 and 1")
            # print(f"DEBUG: Pass rate stats - min: {random_pass_rates.min():.3f}, max: {random_pass_rates.max():.3f}, mean: {random_pass_rates.mean():.3f}")
            
            # Strategy 2: Use the average of the two pass rates during data collection
            self.train_dataset.dataframe["on_policy_pass_rate"] = self.train_dataset.dataframe["qwen2.5_7b_pass_rate"] * 0.5 + self.train_dataset.dataframe["qwen3_30b_pass_rate"] * 0.5

            # Initialize on_policy_avg_length with a default value (e.g., 512 tokens)
            # This could also be computed from existing data if available
            if "on_policy_avg_length" not in self.train_dataset.dataframe.columns:
                self.train_dataset.dataframe["on_policy_avg_length"] = self.config.data.get("max_response_length", 1024*28) / 2 
                print(f"Initialized on_policy_avg_length with default value of {self.config.data.get('max_response_length', 1024*28) / 2} tokens")

            # Sort by pass rate (fixed: was using undefined filtered_df)
            filtered_df = self.train_dataset.dataframe.sort_values(by="on_policy_pass_rate", ascending=False).reset_index(drop=True)
            
            train_dataset_copy = deepcopy(self.train_dataset)
            train_dataset_copy.dataframe = filtered_df

            self.train_dataloader = StatefulDataLoader(
                dataset=train_dataset_copy,
                batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                drop_last=True,
                collate_fn=collate_fn,
                sampler=SequentialSampler(data_source=filtered_df),
            )
            self.n_drop_easy = 0
            self.n_drop_hard = 0

            assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
            return self.train_dataset

        # New filtering strategy: 
        # - pass_rate == 1.0: keep 1/5 randomly
        # - pass_rate == 0.0: keep 1/2 randomly  
        # - others: keep all
        
        original_df = self.train_dataset.dataframe.copy()
        
        # Separate data by pass rate
        perfect_mask = original_df["on_policy_pass_rate"] == 1.0
        failed_mask = original_df["on_policy_pass_rate"] == 0.0
        medium_mask = (original_df["on_policy_pass_rate"] > 0.0) & (original_df["on_policy_pass_rate"] < 1.0)
        
        # Keep all medium difficulty data
        kept_indices = []
        kept_indices.extend(original_df[medium_mask].index.tolist())
        
        # Randomly sample 1/5 of perfect examples
        perfect_indices = original_df[perfect_mask].index.tolist()
        if perfect_indices:
            np.random.seed(42 + epoch_idx)  # Ensure reproducibility but vary by epoch
            n_keep_perfect = max(1, len(perfect_indices) // 4)  # Keep at least 1 if any exist
            n_keep_perfect = min(n_keep_perfect, len(perfect_indices))  # Don't sample more than available
            kept_perfect = np.random.choice(perfect_indices, size=n_keep_perfect, replace=False)
            kept_indices.extend(kept_perfect.tolist())
            self.n_drop_easy = len(perfect_indices) - n_keep_perfect
        
        # Randomly sample 1/2 of failed examples
        failed_indices = original_df[failed_mask].index.tolist()
        if failed_indices:
            np.random.seed(43 + epoch_idx)  # Different seed for failed examples
            n_keep_failed = max(1, len(failed_indices) // 4)  # Keep at least 1 if any exist
            n_keep_failed = min(n_keep_failed, len(failed_indices))  # Don't sample more than available
            kept_failed = np.random.choice(failed_indices, size=n_keep_failed, replace=False)
            kept_indices.extend(kept_failed.tolist())
            self.n_drop_hard = len(failed_indices) - n_keep_failed
        
        # Create filtered dataset
        filtered_df = original_df.loc[kept_indices].reset_index(drop=True)
        
        # Sort by score from high to low
        filtered_df = filtered_df.sort_values(by="on_policy_pass_rate", ascending=False).reset_index(drop=True)
        
        # Create filtered dataset copy
        train_dataset_copy = deepcopy(self.train_dataset)
        train_dataset_copy.dataframe = filtered_df
        
        # Create dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset_copy,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=SequentialSampler(data_source=filtered_df),
        )
        
        # Log filtering statistics
        n_perfect_original = len(perfect_indices)
        n_failed_original = len(failed_indices) 
        n_medium_original = len(original_df[medium_mask])
        n_perfect_kept = len([i for i in kept_indices if i in perfect_indices])
        n_failed_kept = len([i for i in kept_indices if i in failed_indices])
        n_medium_kept = len([i for i in kept_indices if original_df.loc[i, "on_policy_pass_rate"] > 0 and original_df.loc[i, "on_policy_pass_rate"] < 1])
        
        discarded = len(original_df) - len(filtered_df)
        pct = 100 * discarded / len(original_df)

        print(f"Dataset filtering statistics for epoch {epoch_idx}:")
        print(f"Original dataset size: {len(original_df)}")
        print(f"  - Perfect examples (pass_rate=1.0): {n_perfect_original} -> {n_perfect_kept} kept ({n_perfect_kept/max(1,n_perfect_original)*100:.1f}%)")
        print(f"  - Failed examples (pass_rate=0.0): {n_failed_original} -> {n_failed_kept} kept ({n_failed_kept/max(1,n_failed_original)*100:.1f}%)")  
        print(f"  - Medium examples (0<pass_rate<1): {n_medium_original} -> {n_medium_kept} kept (100.0%)")
        print(f"Filtered dataset size: {len(filtered_df)}")
        print(f"Total discarded data points: {discarded}")
        print(f"Total percentage discarded: {pct:.2f}%")

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        return train_dataset_copy

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation on the training data for data filtering
        # self._validate_training_data()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            if self.config.trainer.vary_length:
                train_dataset = self._create_priority_dataloader(epoch_idx=epoch)
            else:
                train_dataset = self.train_dataset
            # create create the default_local_dir if not exists
            if not os.path.exists(self.config.trainer.default_local_dir):
                os.makedirs(self.config.trainer.default_local_dir)
            train_dataset.dataframe.to_csv(os.path.join(self.config.trainer.default_local_dir, 
                f"train_dataset_epoch_{epoch}.csv"), index=False)

            for batch_dict in self.train_dataloader:
                metrics = {}

                if self.config.trainer.vary_length:
                    # Get the average previous pass rate for this batch at the beginning
                    batch_prompt_ids = batch_dict["prompt_id"]
                    unique_prompt_ids = np.unique(batch_prompt_ids)
                    
                    # Get existing pass rates and average lengths from dataset
                    existing_pass_rates = []
                    existing_avg_lengths = []
                    for prompt_id in unique_prompt_ids:
                        row = self.train_dataset.dataframe[self.train_dataset.dataframe['prompt_id'] == prompt_id].iloc[0]
                        existing_pass_rates.append(row['on_policy_pass_rate'])
                        existing_avg_lengths.append(row['on_policy_avg_length'])
                    
                    # Calculate averages for this batch
                    avg_on_policy_pass_rate = np.mean(existing_pass_rates)
                    avg_on_policy_avg_length = np.mean(existing_avg_lengths)
                    max_response_length = self.config.data.get("max_response_length", 1024*28)
                    
                    # Set generation length based on pass rate and average length
                    # High pass rate -> use avg length, low pass rate -> use larger length up to max
                    batch_gen_length = avg_on_policy_avg_length + (max_response_length - avg_on_policy_avg_length) * (1 - avg_on_policy_pass_rate)
                    batch_gen_length = min(batch_gen_length, max_response_length)  # Cap at max response length
                    
                    print(f"Average previous on-policy pass rate for this batch: {avg_on_policy_pass_rate:.4f}")
                    print(f"Average previous on-policy avg length for this batch: {avg_on_policy_avg_length:.1f}")
                    print(f"Batch max gen length: {batch_gen_length:.1f} (max allowed: {max_response_length})")
                    metrics['train/on_policy_pass_rate(mean)'] = avg_on_policy_pass_rate
                    metrics['train/on_policy_avg_length(mean)'] = avg_on_policy_avg_length
                    
                
                # Here the self.train_dataset is the whole dataset, while self.train_dataloader is a
                # DataLoader that yields batches of data across GPUs. 
                # len(self.train_dataloader) * #GPUs = len(self.train_dataset)
                # (bsz, seq_len)
                

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                if self.config.trainer.vary_length:
                    # Set the generation length in meta_info
                    gen_batch.meta_info["target_max_response_length"] = int(batch_gen_length)
                    new_batch.meta_info["target_max_response_length"] = int(batch_gen_length)
                    print(f"Set gen_batch.meta_info['target_max_response_length'] to {int(batch_gen_length)}")
                    metrics['train/target_max_response_length'] = batch_gen_length
                else:
                    gen_batch.meta_info["target_max_response_length"] = int(max_response_length)
                    new_batch.meta_info["target_max_response_length"] = int(max_response_length)
                    metrics['train/target_max_response_length'] = max_response_length

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # (bsz*n, seq_len), interleaved, i.e., [A, B] -> [A, A, A, A, B, B, B, B] for n=4
                    new_batch = new_batch.union(gen_batch_output)

                    new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(new_batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        # TODO(yonghao): logics below should be delayed as late as possible
                        # to maximize overlapping.
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )
                    with _timer('pass_rate_append', timing_raw):
                        # compute the pass rate for the batch 
                        temp_df = pd.DataFrame({
                            "prompt_id": batch.non_tensor_batch["prompt_id"],
                            "on_policy_pass_rate": batch.non_tensor_batch["score"]
                        })
                        pass_rate_df = temp_df.groupby("prompt_id", as_index=False)["on_policy_pass_rate"].mean().set_index('prompt_id')[['on_policy_pass_rate']]
                        
                        # compute the average response length for each prompt_id in the batch
                        # Only include lengths for successful rollouts (score == 1)
                        response_length = batch.batch["responses"].shape[-1]
                        response_mask = batch.batch["attention_mask"][:, -response_length:]
                        response_lengths = response_mask.sum(dim=-1).float().cpu().numpy()  # actual lengths per response
                        scores = batch.non_tensor_batch["score"]
                        
                        # Filter for successful rollouts only (score == 1)
                        successful_mask = scores == 1
                        if np.any(successful_mask):
                            successful_prompt_ids = batch.non_tensor_batch["prompt_id"][successful_mask]
                            successful_response_lengths = response_lengths[successful_mask]
                            
                            temp_length_df = pd.DataFrame({
                                "prompt_id": successful_prompt_ids,
                                "response_length": successful_response_lengths
                            })
                            avg_length_df = temp_length_df.groupby("prompt_id", as_index=False)["response_length"].mean().set_index('prompt_id')[['response_length']]
                            avg_length_df.rename(columns={"response_length": "on_policy_avg_length"}, inplace=True)
                            
                            # Update the dataframe with both pass rates and average lengths
                            self.train_dataset.dataframe = self.train_dataset.dataframe.set_index('prompt_id')
                            self.train_dataset.dataframe.update(pass_rate_df)
                            self.train_dataset.dataframe.update(avg_length_df)
                            self.train_dataset.dataframe = self.train_dataset.dataframe.reset_index()
                        else:
                            # If no successful rollouts in this batch, only update pass rates
                            self.train_dataset.dataframe = self.train_dataset.dataframe.set_index('prompt_id')
                            self.train_dataset.dataframe.update(pass_rate_df)
                            self.train_dataset.dataframe = self.train_dataset.dataframe.reset_index()
                            print("No successful rollouts (score=1) in this batch, skipping length update")

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_difficulty_histogram_metrics(batch=batch, config=self.config))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics['train/num_prompts'] = len(train_dataset.dataframe)
                metrics['train/perct_dropped_prompts'] = 100 * ( (len(self.train_dataset.dataframe) - len(train_dataset.dataframe)) / len(self.train_dataset.dataframe))
                metrics['train/n_drop_easy'] = self.n_drop_easy if self.n_drop_easy is not None else 0
                metrics['train/n_drop_hard'] = self.n_drop_hard if self.n_drop_hard is not None else 0
                metrics['train/epoch'] = epoch
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1