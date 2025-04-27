import os
import argparse
import json
import datasets
from typing import Dict, Any
from transformers import AutoTokenizer

from verl.utils.data_process.utils import set_seed, save_dataset
import numpy as np

def report_prompt_token_stats(tokenizer, dataset, split: str, fmt: str):
    """
    统计 dataset 中所有示例的 prompt token 数量，并打印 min/max/mean/std。
    Assumes each example has a `prompt` 字段：List[{"role":..., "content": str}].
    """
    lengths = []
    for ex in dataset:
        # 只取第一个消息的内容
        content = ex["prompt"][0]["content"]
        # 不加 special tokens
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        lengths.append(len(token_ids))

    lengths = np.array(lengths)
    print(f"[{split} | {fmt}] prompt token count: "
          f"min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, std={lengths.std():.1f}")
# R1‐zero‐style full template

statement_prompt = (
    "Complete the following Lean 4 code with explanatory comments preceding each line of code in markdown format:\n"
    "```lean"
)

def make_map_fn(split: str, data_source: str, prompt_format: str) -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        header    = example.get("header", "")
        statement = example.get("formal_statement", "")
        question  = header + statement

        content = (
            f"{statement_prompt}\n"
            f"{question}\n```\n"
        )

        if prompt_format == "chat_style":
            msg = {"role": "user", "content": content}
            return {
                "data_source": data_source,
                "prompt": [msg],
                "ability": "TheoremProver",
                "statement": question,
                "apply_chat_template": True,
                "response": "",       # placeholder for model output
                "extra_info": {"split": split, "index": idx},
            }
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")
    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess DeepSeek-Prover-V1 and MiniF2F into chat_style."
    )
    parser.add_argument("--data-dir", default="data", help="Base directory to save processed datasets.")
    parser.add_argument("--domain", default="lean4prover", type=str, help="Data domain identifier.")
    parser.add_argument("--name", default="deepseek_prover_v1", type=str, help="Dataset name identifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    base = args.domain
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    # --- 处理 DeepSeek-Prover-V1 的 train 集 ---
    print("Loading DeepSeek-Prover-V1 train dataset...")
    ds_train = datasets.load_dataset("deepseek-ai/DeepSeek-Prover-V1", split="train")
    for fmt in ["chat_style"]:
        ds_name = base
        print(f"\nProcessing format: {fmt} for DeepSeek-Prover-V1 train")
        fn = make_map_fn("train", ds_name, fmt)
        processed = ds_train.map(fn, with_indices=True)
        report_prompt_token_stats(tokenizer, processed, split="train", fmt=fmt)

        # Show and save one sample
        sample = processed[0]
        print(f"Sample (train, {fmt}):")
        print(sample)
        sample_dir = os.path.join(args.data_dir, fmt, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, f"{ds_name}_train_sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)

        out_dir = os.path.join(args.data_dir, fmt, "train")
        os.makedirs(out_dir, exist_ok=True)
        save_path = save_dataset(
            dataset=processed,
            output_dir=out_dir,
            filename_prefix=ds_name,
            sample_size=len(processed)
        )
        print(f"Saved train data ({fmt}) to: {save_path}")

    # --- 处理 MiniF2F 的验证集 ---
    print("\nLoading MiniF2F validation dataset...")
    ds_val = datasets.load_dataset("tonic/MiniF2F", split="train")

    for fmt in ["chat_style"]:
        ds_name = base
        print(f"\nProcessing format: {fmt} for MiniF2F validation")
        fn = make_map_fn("train", ds_name, fmt)
        processed_val = ds_val.map(fn, with_indices=True)
        report_prompt_token_stats(tokenizer, processed_val, split="val", fmt=fmt)

        # Show and save one sample
        sample_val = processed_val[0]
        print(f"Sample (val, {fmt}):")
        print(sample_val)
        sample_val_dir = os.path.join(args.data_dir, fmt, "samples")
        os.makedirs(sample_val_dir, exist_ok=True)
        with open(os.path.join(sample_val_dir, f"{ds_name}_val_sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample_val, f, indent=2, ensure_ascii=False)

        # Save full val dataset
        out_val_dir = os.path.join(args.data_dir, fmt, "val")
        os.makedirs(out_val_dir, exist_ok=True)
        save_path_val = save_dataset(
            dataset=processed_val,
            output_dir=out_val_dir,
            filename_prefix=ds_name,
            sample_size=len(processed_val)
        )
        print(f"Saved val data ({fmt}) to: {save_path_val}")

    print("\nAll done!")
