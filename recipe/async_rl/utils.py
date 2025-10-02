from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np
from verl.trainer.ppo.metric_utils import process_validation_metrics

def compute_validation_metrics(val_data, tokenizer_path):
    batch_metrics = {
        "batch_size": len(val_data),
        "sample_inputs": [],
        "sample_outputs": [], 
        "sample_scores": [],
        "data_sources": [],
        "reward_extra_info": defaultdict(list)
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if "input_ids" in val_data.batch:
            input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in val_data.batch["input_ids"]]
            batch_metrics["sample_inputs"] = input_texts
        if "responses" in val_data.batch:
            output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in val_data.batch["responses"]]
            batch_metrics["sample_outputs"] = output_texts
    except Exception as e:
        print(f"Warning: Could not decode text for validation logging: {e}")

    sequence_scores = None
    if "token_level_scores" in val_data.batch:
        sequence_scores = val_data.batch["token_level_scores"].sum(-1).cpu()
        batch_metrics["sample_scores"] = sequence_scores.tolist()
        
    elif "token_level_rewards" in val_data.batch:
        sequence_scores = val_data.batch["token_level_rewards"].sum(-1).cpu()
        batch_metrics["sample_scores"] = sequence_scores.tolist()
        
    if sequence_scores is not None:
        batch_metrics["reward_extra_info"]["reward"].extend(sequence_scores.tolist())
        
    # Extract data sources
    if "data_source" in val_data.non_tensor_batch:
        batch_metrics["data_sources"] = val_data.non_tensor_batch["data_source"]
    else:
        batch_metrics["data_sources"] = ["unknown"] * len(val_data)

    return batch_metrics

def aggregate_validation_metrics(validation_results):
    if not validation_results:
        return {}

    all_sample_inputs = []
    all_sample_scores = []  
    all_data_sources = []
    all_reward_extra_info = defaultdict(list)
    
    for batch_result in validation_results:
        all_sample_inputs.extend(batch_result.get("sample_inputs", []))
        all_sample_scores.extend(batch_result.get("sample_scores", []))
        all_data_sources.extend(batch_result.get("data_sources", []))
        
        for key, values in batch_result.get("reward_extra_info", {}).items():
            all_reward_extra_info[key].extend(values)

    if not all_sample_scores:
        return {"validation/samples": 0, "validation/avg_reward": 0.0}

    try:
            
        if len(all_data_sources) != len(all_sample_scores):
            all_data_sources = ["unknown"] * len(all_sample_scores)
        if len(all_sample_inputs) != len(all_sample_scores):
            all_sample_inputs = [f"input_{i}" for i in range(len(all_sample_scores))]
            
        reward_extra_infos_dict = dict(all_reward_extra_info)
        if "reward" not in reward_extra_infos_dict:
            reward_extra_infos_dict["reward"] = all_sample_scores
            
        data_sources = np.array(all_data_sources)
        data_src2var2metric2val = process_validation_metrics(
            data_sources, all_sample_inputs, reward_extra_infos_dict
        )
        
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        metric_dict["validation/samples"] = len(all_sample_scores)            
        return metric_dict
        
    except ImportError:
        print("Warning: Could not import process_validation_metrics, using simple metrics")
        
    total_samples = len(all_sample_scores)
    avg_reward = np.mean(all_sample_scores) if all_sample_scores else 0.0
    max_reward = np.max(all_sample_scores) if all_sample_scores else 0.0
    min_reward = np.min(all_sample_scores) if all_sample_scores else 0.0

    return {
        "validation/samples": total_samples,
        "validation/avg_reward": avg_reward,
        "validation/max_reward": max_reward,
        "validation/min_reward": min_reward,
    }