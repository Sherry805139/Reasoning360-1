from datasets import load_dataset, Dataset
from pathlib import Path
import os
import pandas as pd

folder1 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_12k_len_dedup_label_2/"
folder2 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_12k_len_dedup_eval_rl_am_ot_3"

# Create output directory if it doesn't exist
if not os.path.exists(folder2):
    os.makedirs(folder2)

# Define the duplicate columns to check
duplicate_columns = [
    "is_duplicate_within_rl",
    "is_duplicate_within_am", 
    "is_duplicate_within_openthoughts",
    "is_duplicate_within_eval"
]

def post_process(dataset):
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # Check which duplicate columns exist in this dataset
    available_duplicate_cols = [col for col in duplicate_columns if col in df.columns]
    
    if available_duplicate_cols:
        # Create a mask for rows where ANY of the duplicate columns is True
        duplicate_mask = df[available_duplicate_cols].any(axis=1)
        
        # Filter out rows where any duplicate flag is True
        df = df[~duplicate_mask]
    
    # Reset index to avoid duplicate index column issues
    df = df.reset_index(drop=True)
    
    # Convert back to Dataset
    return Dataset.from_pandas(df)

# Create a list to store row counts
row_counts = []

for filename in sorted(Path(folder1).glob("*.parquet")):
    target_file = f"{folder2}/{filename.name}"
    
    # Skip if file already exists in target folder
    if os.path.exists(target_file):
        print(f"Skipping {filename.name} - already exists in target folder")
        continue
    
    # Load and process dataset
    ds = load_dataset("parquet", data_files=str(filename))['train']
    original_rows = len(ds)
    ds = post_process(ds)
    processed_rows = len(ds)
    ds.to_parquet(target_file)
    
    # Record the row count
    row_counts.append(f"{filename.name}: {processed_rows} rows (original: {original_rows})")
    print(f"Processed {filename.name}: {processed_rows} rows (filtered from {original_rows} rows based on duplicate flags)")

# Write row counts to a text file
with open(f"{folder2}/row_counts_after_processing.txt", "w") as f:
    f.write("Row counts after processing (filtering duplicate flags):\n")
    f.write("=" * 60 + "\n")
    for count in row_counts:
        f.write(count + "\n")
    f.write(f"\nTotal files processed: {len(row_counts)}\n")

print(f"\nRow counts saved to: {folder2}/row_counts_after_processing.txt")
    