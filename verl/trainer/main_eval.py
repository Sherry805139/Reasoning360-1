# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""
from collections import defaultdict
import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.reward_score import default_compute_score
from verl.utils.fs import copy_to_local

@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = []
    length_lst = []
    for r in response_lst:
        score = reward_fn(data_source, r, ground_truth, {})  # extra_info is empty for now
        if isinstance(score, dict):
            score = score.get("score", 0.)
        score_lst.append(score)
        # Calculate length - handle both string and list responses
        if isinstance(r, str):
            length_lst.append(len(r))
        elif isinstance(r, list):
            length_lst.append(len(r))
        else:
            length_lst.append(0)
    return data_source, score_lst, np.mean(score_lst), length_lst


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    data_source_all_scores = defaultdict(list)  # Store all individual scores
    data_source_lengths = defaultdict(list)  # Store all response lengths
    compute_score = get_custom_reward_fn(config)

    # need to pull default_compute_score here because we do not have a custom reward fn most often
    if compute_score is None:
        compute_score = default_compute_score

    # Create remote tasks
    remote_tasks = [
        process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) 
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score_list, mean_score, length_list = ray.get(result_id)
                data_source_reward[data_source].append(mean_score)
                data_source_all_scores[data_source].extend(score_list)
                data_source_lengths[data_source].extend(length_list)
                pbar.update(1)

    metric_dict = {}
    detailed_metrics = {}
    
    for data_source, rewards in data_source_reward.items():
        mean_score = np.mean(rewards)
        metric_dict[f"test_score/{data_source}"] = mean_score
        
        # Collect detailed statistics
        all_scores = data_source_all_scores[data_source]
        all_lengths = data_source_lengths[data_source]
        
        detailed_metrics[data_source] = {
            "mean_score": float(mean_score),
            "std_score": float(np.std(rewards)),
            "min_score": float(np.min(rewards)),
            "max_score": float(np.max(rewards)),
            "median_score": float(np.median(rewards)),
            "num_samples": len(rewards),
            "num_total_responses": len(all_scores),
            "all_sample_means": [float(x) for x in rewards],
            "pass_rate": float(np.mean([1.0 if s > 0.5 else 0.0 for s in all_scores])),  # Assuming binary threshold
            "avg_response_length": float(np.mean(all_lengths)),
            "std_response_length": float(np.std(all_lengths)),
            "min_response_length": int(np.min(all_lengths)),
            "max_response_length": int(np.max(all_lengths)),
            "median_response_length": float(np.median(all_lengths))
        }
    
    print(metric_dict)
    
    # =================== Save Results to JSON ===================
    # Determine output path
    input_path = config.data.path
    
    # Create results filename based on input filename
    input_filename = Path(input_path).stem  # Get filename without extension
    output_dir = Path(input_path).parent
    results_filename = f"{input_filename}_eval_results.json"
    results_path = output_dir / results_filename
    
    # Prepare results structure
    results_data = {
        "evaluation_metadata": {
            "input_file": str(input_path),
            "timestamp": datetime.now().isoformat(),
            "total_samples": total,
            "config": {
                "prompt_key": config.data.prompt_key,
                "response_key": config.data.response_key,
                "data_source_key": config.data.data_source_key,
                "reward_model_key": config.data.reward_model_key,
            }
        },
        "summary_metrics": metric_dict,
        "detailed_metrics": detailed_metrics
    }
    
    # Add environment variables for temperature study tracking if available
    if os.getenv("SLURM_ARRAY_TASK_ID"):
        results_data["evaluation_metadata"]["slurm_array_task_id"] = os.getenv("SLURM_ARRAY_TASK_ID")
    if os.getenv("SLURM_JOB_ID"):
        results_data["evaluation_metadata"]["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
    
    # Read temperature from environment or config if available
    if os.getenv("PARAM_STUDY_TEMP"):
        results_data["evaluation_metadata"]["temperature"] = float(os.getenv("PARAM_STUDY_TEMP"))
    
    # Save results to JSON
    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_path}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: Could not save results to {results_path}: {e}")
        # Fallback: try to save to current directory
        fallback_path = Path.cwd() / results_filename
        try:
            with open(fallback_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"Error: Could not save results to fallback location either: {e2}")


if __name__ == "__main__":
    main()