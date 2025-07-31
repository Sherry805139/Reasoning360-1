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

    def _create_priority_dataloader(self, epoch_idx, dynamic_filtering, enable_budget):
        """
        Create the dataloader every time before the epoch starts.
        """
        from torch.utils.data import SequentialSampler
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader
        
        # Initialize columns for the first epoch
        max_easy_ratio = self.config.data.get("max_easy_ratio", 0.1)
        max_hard_ratio = self.config.data.get("max_hard_ratio", 0.2)
        if epoch_idx == 0: 
            # Get the initial pass rate column name from config, with default fallback
            initial_pass_rate_column = self.config.data.get("initial_pass_rate_column", "qwen3_30b_pass_rate")
            self.train_dataset.dataframe["prev_pass_rate"] = self.train_dataset.dataframe[initial_pass_rate_column]
            # use half of the max response length as the average length for the first epoch
            self.train_dataset.dataframe["prev_passed_avg_length"] = self.config.data.get("max_response_length", 1024*28) * 3 / 4 
            self.train_dataset.dataframe["prev_passed_max_length"] = self.config.data.get("max_response_length", 1024*28) * 3 / 4

        
        original_df = self.train_dataset.dataframe.copy()
        
        if dynamic_filtering:
            # Separate data by pass rate
            perfect_mask = original_df["prev_pass_rate"] == 1.0
            failed_mask = original_df["prev_pass_rate"] == 0.0
            medium_mask = (original_df["prev_pass_rate"] > 0.0) & (original_df["prev_pass_rate"] < 1.0)
            
            # Get indices for each category
            medium_indices = original_df[medium_mask].index.tolist()
            perfect_indices = original_df[perfect_mask].index.tolist()
            failed_indices = original_df[failed_mask].index.tolist()
            
            # Keep all medium difficulty data
            kept_indices = set(medium_indices)
            n_medium = len(medium_indices)
            
            # Limit perfect examples to 1/10 of medium examples
            self.n_drop_easy = 0
            if perfect_indices:
                np.random.seed(42 + epoch_idx)
                n_keep_perfect = int(max(1, min(n_medium * max_easy_ratio, len(perfect_indices))))
                if n_keep_perfect > 0:
                    kept_perfect = np.random.choice(perfect_indices, size=n_keep_perfect, replace=False)
                    kept_indices.update(kept_perfect)
                self.n_drop_easy = len(perfect_indices) - n_keep_perfect
            
            # Limit failed examples to 1/5 of medium examples
            self.n_drop_hard = 0
            if failed_indices:
                np.random.seed(43 + epoch_idx)
                n_keep_failed = int(max(1, min(n_medium * max_hard_ratio, len(failed_indices))))
                if n_keep_failed > 0:
                    kept_failed = np.random.choice(failed_indices, size=n_keep_failed, replace=False)
                    kept_indices.update(kept_failed)
                self.n_drop_hard = len(failed_indices) - n_keep_failed
            
            filtered_df = original_df.loc[list(kept_indices)].reset_index(drop=True) 
            # Log filtering statistics
            n_perfect_kept = len(set(perfect_indices) & kept_indices)
            n_failed_kept = len(set(failed_indices) & kept_indices)
            n_medium_kept = len(set(medium_indices) & kept_indices)
            
            print(f"Dataset filtering statistics for epoch {epoch_idx}:")
            print(f"Original dataset size: {len(original_df)}")
            print(f"  - Perfect examples (pass_rate=1.0): {len(perfect_indices)} -> {n_perfect_kept} kept ({n_perfect_kept/max(1,len(perfect_indices))*100:.1f}%)")
            print(f"  - Failed examples (pass_rate=0.0): {len(failed_indices)} -> {n_failed_kept} kept ({n_failed_kept/max(1,len(failed_indices))*100:.1f}%)")  
            print(f"  - Medium examples (0<pass_rate<1): {len(medium_indices)} -> {n_medium_kept} kept ({n_medium_kept/max(1,len(medium_indices))*100:.1f}%)")
            print(f"Filtered dataset size: {len(filtered_df)}")
            print(f"Total discarded data points: {len(original_df) - len(filtered_df)}")
            print(f"Total percentage discarded: {100 * (len(original_df) - len(filtered_df)) / len(original_df):.2f}%")
        else:
            filtered_df = original_df.copy()
        
        def assign_length_budget(row, pass_rate_upper_bound, max_response_length):
            prompt_pass_rate = row['prev_pass_rate']
            passed_prompt_avg_length = row['prev_passed_avg_length']
            passed_prompt_max_length = row['prev_passed_max_length']
            
            # Get configurable multipliers with default values
            perfect_pass_rate_multiplier = self.config.data.get("perfect_pass_rate_multiplier", 1.0)
            high_pass_rate_multiplier = self.config.data.get("high_pass_rate_multiplier", 0.8)
            
            if prompt_pass_rate == 1.0:
                new_length_budget = max(high_pass_rate_multiplier * passed_prompt_max_length, passed_prompt_avg_length)
            elif prompt_pass_rate > pass_rate_upper_bound:
                new_length_budget = max(high_pass_rate_multiplier * passed_prompt_max_length, passed_prompt_avg_length)
            else:
                new_length_budget = passed_prompt_max_length + (max_response_length - passed_prompt_max_length) * (1 - prompt_pass_rate)
            
            new_length_budget = max(new_length_budget, 4000)  # Set minimum to 2000
            new_length_budget = min(new_length_budget, max_response_length)  # Cap at max response length
            
            return int(new_length_budget)
        
        if enable_budget:
            # Shared logic for computing per_prompt_length_budget and creating dataloader
            max_response_length = self.config.data.get("max_response_length", 1024*28)
            pass_rate_upper_bound = self.config.data.get("pass_rate_upper_bound", 1.0)


            filtered_df["per_prompt_length_budget"] = filtered_df.apply(
                lambda row: assign_length_budget(row, pass_rate_upper_bound, max_response_length), axis=1
            )
            filtered_df = filtered_df.sort_values(by="per_prompt_length_budget", ascending=True).reset_index(drop=True)
        else:
            filtered_df["per_prompt_length_budget"] = self.config.data.get("max_response_length", 1024*28)  # Use fixed length budget
        
        # Sort by per_prompt_length_budget for more efficient rollout batching
        filtered_df = filtered_df.sort_values(by="per_prompt_length_budget", ascending=True).reset_index(drop=True)
        
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
            train_dataset = self._create_priority_dataloader(
                epoch_idx=epoch,
                dynamic_filtering=self.config.data.get("dynamic_filtering", False),
                enable_budget=self.config.trainer.get("enable_budget", False),
            )
            # create create the default_local_dir if not exists
            if not os.path.exists(self.config.trainer.default_local_dir):
                os.makedirs(self.config.trainer.default_local_dir)
            train_dataset.dataframe.to_csv(os.path.join(self.config.trainer.default_local_dir, 
                f"train_dataset_epoch_{epoch}.csv"), index=False)

            for batch_dict in self.train_dataloader:
                metrics = {}
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
                
                # Use pre-computed per_prompt_length_budget from the dataframe
                batch_prompt_ids = batch_dict["prompt_id"]
                per_prompt_length_budget = []
                per_prompt_pass_rate = []
                
                for prompt_id in batch_prompt_ids:
                    row = train_dataset.dataframe[train_dataset.dataframe['prompt_id'] == prompt_id].iloc[0]
                    per_prompt_length_budget.append(int(row['per_prompt_length_budget']))
                    per_prompt_pass_rate.append(row['prev_pass_rate'])
                
                # Log statistics
                avg_prompt_pass_rate = np.mean(per_prompt_pass_rate)
                avg_batch_max_length = np.mean(per_prompt_length_budget)
                
                print(f"Average previous on-policy pass rate for this batch: {avg_prompt_pass_rate:.4f}")
                print(f"Average batch max length: {avg_batch_max_length:.1f} (range: {min(per_prompt_length_budget)}-{max(per_prompt_length_budget)})")
                print(f"Set gen_batch.non_tensor_batch['per_prompt_length_budget'] to list of {len(per_prompt_length_budget)} lengths (avg: {avg_batch_max_length:.1f})")
                
                gen_batch.non_tensor_batch["per_prompt_length_budget"] = np.array(per_prompt_length_budget, dtype=object)

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
                    
                    # Fix: Remove per_prompt_length_budget from gen_batch_output if it exists to prevent union conflict
                    if "per_prompt_length_budget" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.non_tensor_batch.pop("per_prompt_length_budget")
                    
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
                    if self.config.trainer.balance_batch and not self.config.trainer.enable_budget:
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
                            "prev_pass_rate": batch.non_tensor_batch["score"]
                        })
                        pass_rate_df = temp_df.groupby("prompt_id", as_index=False)["prev_pass_rate"].mean().set_index('prompt_id')[['prev_pass_rate']]
                        
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
                            avg_length_df.rename(columns={"response_length": "prev_passed_avg_length"}, inplace=True)
                            max_length_df = temp_length_df.groupby("prompt_id", as_index=False)["response_length"].max().set_index('prompt_id')[['response_length']]
                            max_length_df.rename(columns={"response_length": "prev_passed_max_length"}, inplace=True)
                            
                            # Update the dataframe with both pass rates and average lengths
                            self.train_dataset.dataframe = self.train_dataset.dataframe.set_index('prompt_id')
                            self.train_dataset.dataframe.update(pass_rate_df)
                            self.train_dataset.dataframe.update(avg_length_df)
                            self.train_dataset.dataframe.update(max_length_df)
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
                #metrics.update(compute_difficulty_histogram_metrics(batch=batch, config=self.config))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # Add per-prompt metrics for the current batch only
                if hasattr(train_dataset, 'dataframe') and batch is not None:
                    # Get unique prompt_ids from the current batch
                    batch_prompt_ids = batch.non_tensor_batch["prompt_id"]
                    unique_prompt_ids = np.unique(batch_prompt_ids)
                    
                    # Filter dataframe to only include prompts from current batch
                    batch_df = train_dataset.dataframe[train_dataset.dataframe['prompt_id'].isin(unique_prompt_ids)]
                    
                    if len(batch_df) > 0:
                        metrics.update({
                            "train/per_prompt_pass_rate_avg": batch_df["prev_pass_rate"].mean(),
                            "train/per_prompt_pass_rate_std": batch_df["prev_pass_rate"].std(),
                            "train/per_prompt_pass_rate_min": batch_df["prev_pass_rate"].min(),
                            "train/per_prompt_pass_rate_max": batch_df["prev_pass_rate"].max(),
                            "train/num_unique_prompts": len(unique_prompt_ids),
                            "train/per_prompt_length_budget_avg": batch_df["per_prompt_length_budget"].mean(),
                            "train/per_prompt_length_budget_std": batch_df["per_prompt_length_budget"].std(),
                            "train/per_prompt_length_budget_min": batch_df["per_prompt_length_budget"].min(),
                            "train/per_prompt_length_budget_max": batch_df["per_prompt_length_budget"].max(),
                            "train/prev_passed_max_length_avg": batch_df["prev_passed_max_length"].mean(),
                            "train/prev_passed_max_length_std": batch_df["prev_passed_max_length"].std(),
                            "train/prev_passed_max_length_min": batch_df["prev_passed_max_length"].min(),
                            "train/prev_passed_max_length_max": batch_df["prev_passed_max_length"].max(),
                            "train/prev_passed_avg_length_avg": batch_df["prev_passed_avg_length"].mean(),
                            "train/prev_passed_avg_length_std": batch_df["prev_passed_avg_length"].std(),
                            "train/prev_passed_avg_length_min": batch_df["prev_passed_avg_length"].min(),
                            "train/prev_passed_avg_length_max": batch_df["prev_passed_avg_length"].max()
                        })

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

    def _save_dataset_state(self, local_global_step_folder):
        """
        Save the current dataset state including updated pass rates and lengths.
        This is crucial for resuming training with enable_budget feature.
        """
        if not self.config.trainer.get('enable_budget', False):
            return
            
        dataset_state_path = os.path.join(local_global_step_folder, 'dataset_state.pt')
        
        # Save the current dataset state
        dataset_state = {
            'dataframe': self.train_dataset.dataframe.copy(),
            'n_drop_easy': getattr(self, 'n_drop_easy', 0),
            'n_drop_hard': getattr(self, 'n_drop_hard', 0),
        }
        
        torch.save(dataset_state, dataset_state_path)
        print(f"Saved dataset state to {dataset_state_path}")
        print(f"  - Dataset size: {len(self.train_dataset.dataframe)}")
        print(f"  - Pass rate range: {self.train_dataset.dataframe['prev_pass_rate'].min():.3f} - {self.train_dataset.dataframe['prev_pass_rate'].max():.3f}")
        if 'prev_passed_max_length' in self.train_dataset.dataframe.columns:
            print(f"  - Max length range: {self.train_dataset.dataframe['prev_passed_max_length'].min():.1f} - {self.train_dataset.dataframe['prev_passed_max_length'].max():.1f}")

    def _load_dataset_state(self, global_step_folder):
        """
        Load the dataset state including updated pass rates and lengths.
        This restores the learned statistics from previous training.
        """
        if not self.config.trainer.get('enable_budget', False):
            return
            
        dataset_state_path = os.path.join(global_step_folder, 'dataset_state.pt')
        
        if os.path.exists(dataset_state_path):
            print(f"Loading dataset state from {dataset_state_path}")
            dataset_state = torch.load(dataset_state_path, weights_only=False)
            
            # Restore dataset with updated pass rates and lengths
            self.train_dataset.dataframe = dataset_state['dataframe']
            self.n_drop_easy = dataset_state.get('n_drop_easy', 0) 
            self.n_drop_hard = dataset_state.get('n_drop_hard', 0)
            
            print(f"Restored dataset state:")
            print(f"  - Dataset size: {len(self.train_dataset.dataframe)}")
            print(f"  - Pass rate range: {self.train_dataset.dataframe['prev_pass_rate'].min():.3f} - {self.train_dataset.dataframe['prev_pass_rate'].max():.3f}")
            if 'prev_passed_avg_length' in self.train_dataset.dataframe.columns:
                print(f"  - Avg length range: {self.train_dataset.dataframe['prev_passed_avg_length'].min():.1f} - {self.train_dataset.dataframe['prev_passed_avg_length'].max():.1f}")
            if 'prev_passed_max_length' in self.train_dataset.dataframe.columns:
                print(f"  - Max length range: {self.train_dataset.dataframe['prev_passed_max_length'].min():.1f} - {self.train_dataset.dataframe['prev_passed_max_length'].max():.1f}")
        else:
            print(f"No dataset state found at {dataset_state_path}, starting with original dataset")
            self.n_drop_easy = 0
            self.n_drop_hard = 0

    def _save_checkpoint(self):
        """
        Override to include dataset state saving for enable_budget feature.
        """
        # Call parent method to save models and dataloader
        super()._save_checkpoint()
        
        # Save additional dataset state for enable_budget
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        self._save_dataset_state(local_global_step_folder)

    def _load_checkpoint(self):
        """
        Override to include dataset state loading for enable_budget feature.
        """
        # Store original global_steps to detect if we loaded from checkpoint
        original_global_steps = self.global_steps
        
        # Call parent method to load models and dataloader
        result = super()._load_checkpoint()
        
        # If global_steps changed, we loaded from checkpoint
        if self.global_steps > original_global_steps:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            
            # Find the same checkpoint folder that was loaded
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            
            if global_step_folder is not None:
                self._load_dataset_state(global_step_folder)
        
        return result