folder1 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_12k_len_dedup_eval_rl_am_ot_3//"
folder2 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_4/"

from datasets import load_dataset
from pathlib import Path
import os
import pandas as pd

def post_process(dataset):
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # Remove rows where both "qwen3_30b_pass_rate" and "qwen2.5_7b_pass_rate" exist 
    # and qwen2.5_7b_pass_rate > qwen3_30b_pass_rate
    if "qwen3_30b_pass_rate" in df.columns and "qwen2.5_7b_pass_rate" in df.columns:
        # Create mask for rows to keep (opposite of condition to remove)
        mask = ~((df["qwen3_30b_pass_rate"].notna()) & 
                 (df["qwen2.5_7b_pass_rate"].notna()) & 
                 (df["qwen2.5_7b_pass_rate"] > df["qwen3_30b_pass_rate"]) &
                 (df["qwen2.5_7b_pass_rate"] > df["deepseek_r1_0528_pass_rate"]))
        df = df[mask]
    
    # Reset index to avoid duplicate index columns issue
    df = df.reset_index(drop=True)
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_pandas(df)

if not os.path.exists(folder2):
    os.makedirs(folder2)

# Create a list to store row counts
row_counts = []

for filename in sorted(Path(folder1).glob("*.parquet")):
    target_file = f"{folder2}/{filename.name}"
    
    # Skip if file already exists in target folder
    if os.path.exists(target_file):
        print(f"Skipping {filename.name} - already exists in target folder")
        continue
    
    # Option 1: Save as parquet file (no train split structure)
    ds = load_dataset("parquet", data_files=str(filename))['train']
    original_rows = len(ds)
    ds = post_process(ds)
    processed_rows = len(ds)
    ds.to_parquet(target_file)
    
    # Record the row count
    row_counts.append(f"{filename.name}: {processed_rows} rows (original: {original_rows})")
    print(f"Processed {filename.name}: {processed_rows} rows (removed {original_rows - processed_rows} rows)")

# Write row counts to a text file
with open(f"{folder2}/row_counts_after_processing.txt", "w") as f:
    f.write("Row counts after processing (removing flipscore rows):\n")
    f.write("=" * 60 + "\n")
    for count in row_counts:
        f.write(count + "\n")
    f.write(f"\nTotal files processed: {len(row_counts)}\n")

print(f"\nRow counts saved to: {folder2}/row_counts_after_processing.txt")
