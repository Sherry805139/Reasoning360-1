# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Preprocess the DeepSeek-Prover-V1 dataset into two prompt styles and showcase one sample from each:

1. **chat_style** (for instruct-tuned models)
2. **r1_zero_style** (for base models)

Outputs are saved under `data_dir/{format}/train/` as parquet, and one example per style is printed.
"""
import os
import argparse
import json
import datasets
from typing import Dict, Any

from verl.utils.data_process.utils import set_seed, save_dataset

# R1-zero-style full template
statement_prompt = "Here is the Lean4 theorem statement. You need to complete the full proof following the statement: "
        # Combine statement prompt with question and math instruction

def make_map_fn(split: str, data_source: str, prompt_format: str) -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:

        header = example.get("header", "")
        statement = example.get("formal_statement", "")
        goal = example.get("goal", "")
        proof = example.get("formal_proof", "")
        question = header + statement + goal
        response = (question+proof).replace("\\n", "\n")

        content = (
            statement_prompt + "" + question +
            " Please output the final answer within \\boxed{}."
        )
        if prompt_format == "chat_style":
            # chat-style prompt
            msg = {"role": "user", "content": content}
            return {
                "data_source": data_source,
                "prompt": [msg],
                "ability": "TheoremProver",
                "statement": question,
                "apply_chat_template": True,
                "response": response,
                "extra_info": {"split": split, "index": idx},
            }
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DeepSeek-Prover-V1 into chat and r1_zero styles.")
    parser.add_argument("--data-dir", default="data", help="Base directory to save processed datasets.")
    parser.add_argument("--domain", default="prover", type=str, help="Data domain identifier.")
    parser.add_argument("--name", default="deepseek_prover_v1", type=str, help="Dataset name identifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    base = f"{args.domain}"
    print("Loading DeepSeek-Prover-V1 dataset...")
    dataset = datasets.load_dataset("deepseek-ai/DeepSeek-Prover-V1", split="train")

    for fmt in ["chat_style"]:
        ds_name = f"{base}__{fmt}"
        print(f"\nProcessing format: {fmt}")
        fn = make_map_fn("train", ds_name, fmt)
        processed = dataset.map(fn, with_indices=True)

        # Show one sample
        sample = processed[0]
        print(f"Sample for {fmt}:")
        print(sample)

        # Save full dataset
        out_dir = os.path.join(args.data_dir, fmt, "train")
        os.makedirs(out_dir, exist_ok=True)
        save_path = save_dataset(
            dataset=processed,
            output_dir=out_dir,
            filename_prefix=ds_name,
            sample_size=len(processed)
        )
        print(f"Saved {fmt} data to: {save_path}")
    print("All done!")
