folder1 = "/mnt/sharefs/users/haonan.li/data/k2/train_dedup_am_12k_len"
folder2 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len"

from datasets import load_dataset
from pathlib import Path
import os
import pandas as pd

def post_process(dataset):
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # remove column "model_pass_rate" "detailed_scores" "scores"
    df = df.drop(columns=["model_pass_rate", "detailed_scores", "scores"])
    # rename column "pass_rate" to "r1_0528_pass_rate"
    df = df.rename(columns={"pass_rate": "deepseek_r1_0528_pass_rate"})
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_pandas(df)

if not os.path.exists(folder2):
    os.makedirs(folder2)

for filename in sorted(Path(folder1).glob("*.parquet")):
    # Option 1: Save as parquet file (no train split structure)
    ds = load_dataset("parquet", data_files=str(filename))['train']
    ds = post_process(ds)
    ds.to_parquet(f"{folder2}/{filename.name}")
    
    # Option 2: Save maintaining original structure with train split
    # Uncomment the lines below if you want to maintain the Dataset structure
    # full_dataset = load_dataset("parquet", data_files=str(filename))
    # processed_train = post_process(full_dataset['train'])
    # from datasets import DatasetDict
    # new_dataset = DatasetDict({'train': processed_train})
    # new_dataset.save_to_disk(f"{folder2}/{filename.stem}")
