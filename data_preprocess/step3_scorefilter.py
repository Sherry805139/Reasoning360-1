folder1 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_4/"
folder2 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_1/"
folder3 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_2/"
folder4 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_3/"

# 5_1: remove rows where deepseek_r1_0528_pass_rate is 0 and 1 (diff_range=middle)
# 5_2: remove rows where deepseek_r1_0528_pass_rate is 0, keep at most 50% of final dataset size as rows with deepseek_r1_0528_pass_rate is 1 (diff_range=easy)
# 5_3: keep at most 10% of final dataset size as rows with deepseek_r1_0528_pass_rate is 1 (diff_range=wide)


from datasets import load_dataset
from pathlib import Path
import os
import pandas as pd

if not os.path.exists(folder2):
    os.makedirs(folder2)
if not os.path.exists(folder3):
    os.makedirs(folder3)
if not os.path.exists(folder4):
    os.makedirs(folder4)

def post_process_method_5_1(dataset):
    """
    Method 5_1: remove rows where deepseek_r1_0528_pass_rate is 0 and 1 (diff=middle)
    This keeps only rows where deepseek_r1_0528_pass_rate is not 0 and not 1
    """
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # Filter based on deepseek_r1_0528_pass_rate
    if 'deepseek_r1_0528_pass_rate' in df.columns:
        # Keep rows where deepseek_r1_0528_pass_rate is neither 0 nor 1
        filtered_df = df[(df['deepseek_r1_0528_pass_rate'] != 0) & (df['deepseek_r1_0528_pass_rate'] != 1)]
        df = filtered_df
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_pandas(df)


def post_process_method_5_2(dataset):
    """
    Method 5_2: remove rows where deepseek_r1_0528_pass_rate is 0, 
    keep at most 50% of the final dataset size as rows with deepseek_r1_0528_pass_rate is 1 (diff=easy)
    """
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    if 'deepseek_r1_0528_pass_rate' in df.columns:
        # Remove rows where deepseek_r1_0528_pass_rate is 0
        df_no_zeros = df[df['deepseek_r1_0528_pass_rate'] != 0]
        
        # Separate rows with pass_rate = 1 and others
        rows_with_1 = df_no_zeros[df_no_zeros['deepseek_r1_0528_pass_rate'] == 1]
        other_rows = df_no_zeros[df_no_zeros['deepseek_r1_0528_pass_rate'] != 1]
        
        # Calculate how many rows with pass_rate = 1 we can keep
        # We want at most 50% of the final dataset to be rows with pass_rate = 1
        # So: rows_with_1_to_keep + len(other_rows) = final_size
        # And: rows_with_1_to_keep <= 0.5 * final_size
        # Therefore: rows_with_1_to_keep <= 0.5 * (rows_with_1_to_keep + len(other_rows))
        # Solving: rows_with_1_to_keep <= len(other_rows)
        max_rows_with_1 = len(other_rows)  # This ensures 50% of final dataset
        
        if len(rows_with_1) > 0:
            sample_size = min(len(rows_with_1), max_rows_with_1)
            rows_with_1_sampled = rows_with_1.sample(n=sample_size, random_state=42)
        else:
            rows_with_1_sampled = rows_with_1
        
        # Combine the filtered data
        df = pd.concat([other_rows, rows_with_1_sampled], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_pandas(df)


def post_process_method_5_3(dataset):
    """
    Method 5_3: keep at most 10% of the final dataset size as rows with deepseek_r1_0528_pass_rate is 1 (diff=hard)
    """
    # Convert Dataset to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    if 'deepseek_r1_0528_pass_rate' in df.columns:
        # Separate rows with pass_rate = 1 and others
        rows_with_1 = df[df['deepseek_r1_0528_pass_rate'] == 1]
        other_rows = df[df['deepseek_r1_0528_pass_rate'] != 1]
        
        # Calculate how many rows with pass_rate = 1 we can keep
        # We want at most 10% of the final dataset to be rows with pass_rate = 1
        # So: rows_with_1_to_keep + len(other_rows) = final_size
        # And: rows_with_1_to_keep <= 0.1 * final_size
        # Therefore: rows_with_1_to_keep <= 0.1 * (rows_with_1_to_keep + len(other_rows))
        # Solving: rows_with_1_to_keep <= len(other_rows) / 9
        max_rows_with_1 = max(1, len(other_rows) // 9)  # This ensures 10% of final dataset
        
        if len(rows_with_1) > 0:
            sample_size = min(len(rows_with_1), max_rows_with_1)
            rows_with_1_sampled = rows_with_1.sample(n=sample_size, random_state=42)
        else:
            rows_with_1_sampled = rows_with_1
        
        # Combine the filtered data
        df = pd.concat([other_rows, rows_with_1_sampled], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_pandas(df)


def process_folder(source_folder, target_folder, post_process_method, method_name):
    """Process all parquet files in source_folder using the specified post_process_method"""
    row_counts = []
    
    for filename in sorted(Path(source_folder).glob("*.parquet")):
        # # to process a single file
        # if filename.name != "ifbench__fixed_85.6k.parquet":
        #     continue
        # target_file = f"{target_folder}/{filename.name}"
        
        # Skip if file already exists in target folder
        if os.path.exists(target_file):
            print(f"Skipping {filename.name} - already exists in target folder")
            continue
        
        # Load and process the dataset
        ds = load_dataset("parquet", data_files=str(filename))['train']
        original_rows = len(ds)
        ds = post_process_method(ds)
        processed_rows = len(ds)
        ds.to_parquet(target_file)
        
        # Record the row count
        row_counts.append(f"{filename.name}: {processed_rows} rows (original: {original_rows})")
        print(f"Processed {filename.name}: {processed_rows} rows (filtered from {original_rows} rows using {method_name})")
    
    # Write row counts to a text file
    with open(f"{target_folder}/row_counts_after_processing.txt", "w") as f:
        f.write(f"Row counts after processing using {method_name}:\n")
        f.write("=" * 60 + "\n")
        for count in row_counts:
            f.write(count + "\n")
        f.write(f"\nTotal files processed: {len(row_counts)}\n")
    
    print(f"\nRow counts saved to: {target_folder}/row_counts_after_processing.txt")
    return row_counts

# Process each folder with its corresponding method
print("Processing folder2 with method 5_1 (remove pass_rate 0 and 1)...")
process_folder(folder1, folder2, post_process_method_5_1, "method_5_1")

print("\nProcessing folder3 with method 5_2 (remove pass_rate 0, keep 50% of pass_rate 1)...")
process_folder(folder1, folder3, post_process_method_5_2, "method_5_2")

print("\nProcessing folder4 with method 5_3 (keep 10% of pass_rate 1)...")
process_folder(folder1, folder4, post_process_method_5_3, "method_5_3")

print("\nAll processing completed!")
