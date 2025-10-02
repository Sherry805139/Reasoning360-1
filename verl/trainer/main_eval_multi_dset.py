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
Offline evaluate the performance of generated files using reward model and ground truth verifier.
Supports both single file and batch evaluation of multiple files.
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
import glob

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.reward_score import default_compute_score
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = []
    length_lst = []
    for r in response_lst:
        score = reward_fn(data_source, r, ground_truth, {})
        if isinstance(score, dict):
            score = score.get("score", 0.)
        score_lst.append(score)
        # Calculate length
        if isinstance(r, str):
            length_lst.append(len(r))
        elif isinstance(r, list):
            length_lst.append(len(r))
        else:
            length_lst.append(0)
    return data_source, score_lst, np.mean(score_lst), length_lst


def evaluate_single_file(file_path, config, compute_score):
    """Evaluate a single parquet file and return results"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {file_path}")
    print(f"{'='*60}\n")
    
    local_path = copy_to_local(file_path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    total = len(dataset)

    data_source_reward = defaultdict(list)
    data_source_all_scores = defaultdict(list)
    data_source_lengths = defaultdict(list)

    # Create remote tasks
    remote_tasks = [
        process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) 
        for i in range(total)
    ]

    # Process results
    with tqdm(total=total, desc=f"Evaluating {Path(file_path).name}") as pbar:
        while len(remote_tasks) > 0:
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
            "pass_rate": float(np.mean([1.0 if s > 0.5 else 0.0 for s in all_scores])),
            "avg_response_length": float(np.mean(all_lengths)),
            "std_response_length": float(np.std(all_lengths)),
            "min_response_length": int(np.min(all_lengths)),
            "max_response_length": int(np.max(all_lengths)),
            "median_response_length": float(np.median(all_lengths))
        }
    
    print(f"\nResults for {Path(file_path).name}:")
    print(json.dumps(metric_dict, indent=2))
    
    return metric_dict, detailed_metrics


def save_results(file_path, metric_dict, detailed_metrics, config):
    """Save evaluation results to JSON file"""
    input_filename = Path(file_path).stem
    output_dir = Path(file_path).parent
    results_filename = f"{input_filename}_eval_results.json"
    results_path = output_dir / results_filename
    
    results_data = {
        "evaluation_metadata": {
            "input_file": str(file_path),
            "timestamp": datetime.now().isoformat(),
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
    
    # Add environment variables for temperature study tracking
    if os.getenv("SLURM_ARRAY_TASK_ID"):
        results_data["evaluation_metadata"]["slurm_array_task_id"] = os.getenv("SLURM_ARRAY_TASK_ID")
    if os.getenv("SLURM_JOB_ID"):
        results_data["evaluation_metadata"]["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
    if os.getenv("PARAM_STUDY_TEMP"):
        results_data["evaluation_metadata"]["temperature"] = float(os.getenv("PARAM_STUDY_TEMP"))
    
    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {results_path}")
        return results_path
    except Exception as e:
        print(f"Warning: Could not save results to {results_path}: {e}")
        fallback_path = Path.cwd() / results_filename
        try:
            with open(fallback_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to fallback location: {fallback_path}")
            return fallback_path
        except Exception as e2:
            print(f"Error: Could not save results to fallback location either: {e2}")
            return None


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    compute_score = get_custom_reward_fn(config)
    if compute_score is None:
        compute_score = default_compute_score

    # Check if we're in batch mode or single file mode
    if hasattr(config.data, 'paths') and config.data.paths:
        # Batch mode - multiple files
        if isinstance(config.data.paths, str):
            # Could be comma-separated string
            file_paths = [p.strip() for p in config.data.paths.split(',')]
        else:
            file_paths = list(config.data.paths)
        
        print(f"\n{'='*60}")
        print(f"Batch Evaluation Mode")
        print(f"Evaluating {len(file_paths)} files")
        print(f"{'='*60}\n")
        
        all_results = {}
        
        for file_path in file_paths:
            try:
                metric_dict, detailed_metrics = evaluate_single_file(file_path, config, compute_score)
                results_path = save_results(file_path, metric_dict, detailed_metrics, config)
                
                file_name = Path(file_path).name
                all_results[file_name] = {
                    "metrics": metric_dict,
                    "detailed_metrics": detailed_metrics,
                    "results_file": str(results_path) if results_path else None
                }
            except Exception as e:
                print(f"Error evaluating {file_path}: {e}")
                all_results[Path(file_path).name] = {"error": str(e)}
        
        # Save aggregated results summary
        if hasattr(config.data, 'batch_results_path'):
            batch_results_path = config.data.batch_results_path
        else:
            # Default location
            batch_results_path = Path(file_paths[0]).parent / "batch_evaluation_summary.json"
        
        batch_summary = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_files": len(file_paths),
                "files": file_paths
            },
            "results": all_results
        }
        
        try:
            with open(batch_results_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            print(f"\n{'='*60}")
            print(f"Batch evaluation summary saved to: {batch_results_path}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not save batch summary: {e}")
        
    elif hasattr(config.data, 'path_pattern') and config.data.path_pattern:
        # Pattern-based batch mode - evaluate all files matching pattern
        pattern = config.data.path_pattern
        file_paths = glob.glob(pattern)
        
        if not file_paths:
            print(f"No files found matching pattern: {pattern}")
            return
        
        print(f"\n{'='*60}")
        print(f"Pattern-Based Batch Evaluation Mode")
        print(f"Pattern: {pattern}")
        print(f"Found {len(file_paths)} files")
        print(f"{'='*60}\n")
        
        all_results = {}
        
        for file_path in file_paths:
            try:
                metric_dict, detailed_metrics = evaluate_single_file(file_path, config, compute_score)
                results_path = save_results(file_path, metric_dict, detailed_metrics, config)
                
                file_name = Path(file_path).name
                all_results[file_name] = {
                    "metrics": metric_dict,
                    "detailed_metrics": detailed_metrics,
                    "results_file": str(results_path) if results_path else None
                }
            except Exception as e:
                print(f"Error evaluating {file_path}: {e}")
                all_results[Path(file_path).name] = {"error": str(e)}
        
        # Save aggregated results
        base_dir = Path(file_paths[0]).parent
        batch_results_path = base_dir / "pattern_evaluation_summary.json"
        
        batch_summary = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "pattern": pattern,
                "num_files": len(file_paths),
                "files": file_paths
            },
            "results": all_results
        }
        
        try:
            with open(batch_results_path, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            print(f"\n{'='*60}")
            print(f"Pattern evaluation summary saved to: {batch_results_path}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Warning: Could not save pattern summary: {e}")
    
    else:
        # Single file mode (backward compatibility)
        print("Single File Evaluation Mode")
        metric_dict, detailed_metrics = evaluate_single_file(config.data.path, config, compute_score)
        save_results(config.data.path, metric_dict, detailed_metrics, config)


if __name__ == "__main__":
    main()