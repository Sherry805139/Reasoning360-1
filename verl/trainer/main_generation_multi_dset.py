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
Generate responses given multiple datasets of prompts with different configurations
"""

import json
import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def merge_responses(responses):
    """Merge multiple response lists into one"""
    merged = []
    for r in responses:
        merged.extend(r)
    return merged


def extract_content(p):
    """Extract content from prompt (handle both string and list formats)"""
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            return p
    if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
        return p[0].get("content", "")
    return str(p)


def merge_aime_responses(dataset, output_lst, prompt_key="prompt", response_key="responses"):
    """Merge responses for AIME dataset based on prompt content"""
    if hasattr(dataset, "to_pandas"):  # polars DataFrame
        df = dataset.to_pandas()
        is_polars_df = True
    else:
        df = dataset.copy()
        is_polars_df = False

    df[response_key] = output_lst
    df["prompt_content"] = df[prompt_key].apply(extract_content)

    group_keys = ["prompt_content"]
    agg_dict = {response_key: merge_responses}

    for col in df.columns:
        if col not in group_keys + [response_key]:
            agg_dict[col] = "first"

    df_merged = df.groupby(group_keys, as_index=False).agg(agg_dict)

    if is_polars_df:
        import polars as pl
        return pl.DataFrame(df_merged)
    else:
        return df_merged


def process_single_dataset(
    wg,
    tokenizer,
    dataset,
    dataset_name,
    output_path,
    dataset_config,
    prompt_key="prompt"
):
    """Process a single dataset with its specific configuration"""

    import torch
    import gc
    
    # Clear GPU cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Configuration: {dataset_config}")
    print(f"{'='*60}\n")
    
    # Extract configuration
    n_samples = dataset_config['n_samples']
    batch_size = dataset_config['batch_size']
    prompt_length = dataset_config['prompt_length']
    response_length = dataset_config['response_length']
    
    # Read dataset
    is_polars_df = False
    if "livecodebench" in output_path.lower():
        import polars as pl
        ds = pl.read_parquet(dataset)
        chat_lst = list(ds[prompt_key])
        chat_lst = [list(chat) for chat in chat_lst]
        ground_truth_lst = list(ds["reward_model"])
        is_polars_df = True
    else:
        ds = pd.read_parquet(dataset)
        chat_lst = ds[prompt_key].tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]
        ground_truth_lst = ds["reward_model"].tolist()

    # Handle n_samples
    if n_samples > 1:
        chat_lst = chat_lst * n_samples
        ground_truth_lst = ground_truth_lst * n_samples

    total_samples = len(chat_lst)
    num_batch = -(-total_samples // batch_size)

    output_lst = []

    for batch_idx in range(num_batch):
        print(f"[{dataset_name}] [{batch_idx + 1}/{num_batch}] Processing batch")
        
        batch_chat_lst = chat_lst[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        print(f"[{dataset_name}] [{batch_idx + 1}/{num_batch}] Generating responses")
        
        # Update generation config for this dataset
        wg.update_rollout_config(
            prompt_length=prompt_length,
            response_length=response_length
        )
        
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)
        
        output_texts = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_len = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_len:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_texts.append(response_str)

        # Remove padding
        pad_token = tokenizer.pad_token
        output_text_unpad = []
        for text in output_texts:
            output_text_unpad.append(text.replace(pad_token, ""))

        output_lst.extend(output_text_unpad)

    # Convert output_lst from (n_samples * n_data,) to (n_data, n_samples)
    original_data_size = len(ds)
    output_lst = np.array(output_lst).reshape(n_samples, original_data_size)
    output_lst = output_lst.T.tolist()

    original_chat_lst = chat_lst[:original_data_size]
    original_ground_truth_lst = ground_truth_lst[:original_data_size]

    # Check if we should merge AIME responses
    should_merge_aime = "aime" in output_path.lower()
    ds["responses"] = output_lst

    if should_merge_aime:
        print(f"[{dataset_name}] Merging AIME responses by prompt content")
        merged_dataset = merge_aime_responses(ds, output_lst, prompt_key, "responses")

        output_dir = os.path.dirname(output_path)
        makedirs(output_dir, exist_ok=True)

        if hasattr(merged_dataset, "write_parquet"):
            merged_dataset.write_parquet(output_path)
        else:
            merged_dataset.to_parquet(output_path)

        print(f"[{dataset_name}] Saved merged responses to {output_path}")
    else:
        if is_polars_df:
            import polars as pl
            ds = ds.with_columns(pl.Series("responses", output_lst))
            output_dir = os.path.dirname(output_path)
            makedirs(output_dir, exist_ok=True)
            ds.write_parquet(output_path)
        else:
            ds["responses"] = output_lst
            output_dir = os.path.dirname(output_path)
            makedirs(output_dir, exist_ok=True)
            ds.to_parquet(output_path)

        # Save JSON results
        result_list = [
            {
                "prompt": chat,
                "response": output,
                "ground_truth": str(ground_truth),
            }
            for chat, output, ground_truth in zip(original_chat_lst, output_lst, original_ground_truth_lst)
        ]
        
        json_path = output_path.replace(".parquet", f"_{dataset_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=2, ensure_ascii=False)
    
    print(f"[{dataset_name}] Completed processing\n")
    return True


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Initialize tokenizer
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if "olmoe" in local_path.lower() and "instruct" not in local_path.lower():
        tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|system|>\\n' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|user|>\\n' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{% if not loop.last %}"
            "{{ '<|assistant|>\\n'  + message['content'] + eos_token + '\\n' }}"
            "{% else %}"
            "{{ '<|assistant|>\\n'  + message['content'] + eos_token }}"
            "{% endif %}"
            "{% endif %}"
            "{% if loop.last and add_generation_prompt %}"
            "{{ '<|assistant|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize worker group
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    # Parse multiple dataset paths
    if hasattr(config.data, 'paths') and config.data.paths:
        # Multi-dataset mode
        dataset_paths = config.data.paths.split(',')
        dataset_names = config.data.dataset_names.split(',')
        output_paths = config.data.output_paths.split(',')
        
        # Load dataset configurations
        with open(config.data.dataset_config_file, 'r') as f:
            dataset_configs = json.load(f)
        
        print(f"\n{'='*60}")
        print("Multi-dataset generation mode")
        print(f"Processing {len(dataset_paths)} datasets")
        print(f"{'='*60}\n")
        
        # Process each dataset
        for dataset_path, dataset_name, output_path in zip(dataset_paths, dataset_names, output_paths):
            if dataset_name not in dataset_configs:
                print(f"WARNING: No configuration found for {dataset_name}, skipping")
                continue
            
            dataset_config = dataset_configs[dataset_name]
            
            # Update temperature for this generation
            if config.rollout.temperature == 0.0:
                assert dataset_config['n_samples'] == 1, "When temperature=0, n_samples must be 1."
            
            process_single_dataset(
                wg=wg,
                tokenizer=tokenizer,
                dataset=dataset_path.strip(),
                dataset_name=dataset_name.strip(),
                output_path=output_path.strip(),
                dataset_config=dataset_config,
                prompt_key=config.data.prompt_key
            )
        
        print(f"\n{'='*60}")
        print("All datasets processed successfully!")
        print(f"{'='*60}\n")
        
    else:
        # Single dataset mode (backward compatibility)
        print(f"\n{'='*60}")
        print("Single dataset mode")
        print(f"{'='*60}\n")
        
        dataset_config = {
            'n_samples': config.data.n_samples,
            'batch_size': config.data.batch_size,
            'prompt_length': config.rollout.prompt_length,
            'response_length': config.rollout.response_length,
        }
        
        if config.rollout.temperature == 0.0:
            assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
        
        process_single_dataset(
            wg=wg,
            tokenizer=tokenizer,
            dataset=config.data.path,
            dataset_name=os.path.basename(config.data.path),
            output_path=config.data.output_path,
            dataset_config=dataset_config,
            prompt_key=config.data.prompt_key
        )
        
        print(f"\n{'='*60}")
        print("Dataset processed successfully!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()