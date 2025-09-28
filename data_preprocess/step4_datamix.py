folder1 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_1"
folder2 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_2"
folder3 = "/mnt/sharefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_3"

from datasets import load_dataset
from pathlib import Path
import os
import pandas as pd
import shutil

def create_datamix_folders():
    """Create new folders with _datamix_6 suffix for each source folder"""
    folders = [folder1, folder2, folder3]
    target_folders = []
    
    for folder in folders:
        target_folder = folder + "_datamix_6"
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"Created folder: {target_folder}")
        else:
            print(f"Folder already exists: {target_folder}")
        target_folders.append(target_folder)
    
    return target_folders

def get_math_total_rows(source_folder):
    """Get total rows from math files"""
    math_files = [
        "math__combined_118.2k.part1.parquet",
        "math__combined_118.2k.part2.parquet"
    ]
    
    total_rows = 0
    for math_file in math_files:
        file_path = os.path.join(source_folder, math_file)
        if os.path.exists(file_path):
            ds = load_dataset("parquet", data_files=file_path)['train']
            total_rows += len(ds)
            print(f"  {math_file}: {len(ds)} rows")
    
    return total_rows

def copy_math_files(source_folder, target_folder):
    """Copy math files and log total rows"""
    math_files = [
        "math__combined_118.2k.part1.parquet",
        "math__combined_118.2k.part2.parquet"
    ]
    
    total_rows = 0
    for math_file in math_files:
        source_path = os.path.join(source_folder, math_file)
        target_path = os.path.join(target_folder, math_file)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            ds = load_dataset("parquet", data_files=source_path)['train']
            rows = len(ds)
            total_rows += rows
            print(f"  Copied {math_file}: {rows} rows")
        else:
            print(f"  Warning: {math_file} not found in {source_folder}")
    
    print(f"  Total math rows: {total_rows}")
    return total_rows

def process_all_files(source_folder, target_folder, math_total_rows):
    """Process specific files with restrictions and copy remaining files"""
    # Define math files and files to process with specific percentages
    math_files = [
        "math__combined_118.2k.part1.parquet",
        "math__combined_118.2k.part2.parquet"
    ]
    
    files_to_process = {
        "stem__web_31.7k.parquet": 0.3,
        "stem__nemotron_13.3k.parquet": 0.3,
        "simulation__codeio_fixed_12.1k.parquet": 0.0,
        "logic__reasoning_gym_40.6k.parquet": 0.3,
        "logic__synlogic_12.1k.parquet": 0.3
    }
    
    # Get all parquet files in source folder
    all_files = [f for f in os.listdir(source_folder) if f.endswith('.parquet')]
    remaining_files = [f for f in all_files if f not in math_files and f not in files_to_process.keys()]
    
    # Process specific files with restrictions
    print("  Processing specific files with restrictions...")
    for filename, percentage in files_to_process.items():
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        
        if os.path.exists(source_path):
            ds = load_dataset("parquet", data_files=source_path)['train']
            original_rows = len(ds)
            
            if percentage == 0.0:
                # For 0.0 percentage, copy all rows
                print(f"  {filename}: copied all {original_rows} rows (0% limit = copy all)")
                ds.to_parquet(target_path)
            else:
                # Calculate max rows based on percentage of math total rows
                max_rows = int(math_total_rows * percentage)
                print(f"  {filename}: max rows allowed ({percentage*100}% of math): {max_rows}")
                
                if original_rows > max_rows:
                    # Sample the required number of rows
                    ds = ds.select(range(max_rows))
                    print(f"  {filename}: sampled {max_rows} rows from {original_rows}")
                else:
                    print(f"  {filename}: kept all {original_rows} rows (within limit)")
                
                ds.to_parquet(target_path)
        else:
            print(f"  Warning: {filename} not found in {source_folder}")
    
    # Copy remaining files
    print(f"  Found {len(remaining_files)} remaining files to copy")
    for filename in remaining_files:
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"  Copied {filename}")
        else:
            print(f"  Warning: {filename} not found in {source_folder}")

def main():
    """Main function to process all folders"""
    print("Creating datamix folders...")
    target_folders = create_datamix_folders()
    
    source_folders = [folder1, folder2, folder3]
    
    for i, (source_folder, target_folder) in enumerate(zip(source_folders, target_folders), 1):
        print(f"\nProcessing folder {i}: {source_folder}")
        print(f"Target folder: {target_folder}")
        
        # Step 1: Copy math files and log total rows
        print("Copying math files...")
        math_total_rows = copy_math_files(source_folder, target_folder)
        
        # Step 2: Process all other files (specific restrictions + copy remaining)
        print("Processing all other files...")
        process_all_files(source_folder, target_folder, math_total_rows)
        
        print(f"Completed processing folder {i}")

if __name__ == "__main__":
    main()
