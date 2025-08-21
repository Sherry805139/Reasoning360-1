#!/usr/bin/env python3
"""
Visualize pass rate and response length distributions from scored parquet files.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """Read a parquet file using HuggingFace datasets."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame()
    
    dataset = Dataset.from_parquet(file_path)
    return dataset.to_pandas()


def extract_pass_rates(df: pd.DataFrame) -> List[float]:
    """Extract pass rates from DataFrame."""
    pass_rates = []
    for _, row in df.iterrows():
        if 'pass_rate' in row and row['pass_rate'] is not None:
            pass_rates.append(row['pass_rate'])
    return pass_rates


def extract_response_lengths(df: pd.DataFrame) -> List[int]:
    """Extract all response lengths from DataFrame, filtering to 32k limit."""
    all_lengths = []
    filtered_count = 0
    total_count = 0
    
    for _, row in df.iterrows():
        if 'response_lengths' in row and row['response_lengths'] is not None:
            lengths = row['response_lengths']
            if isinstance(lengths, (list, np.ndarray)):
                for length in lengths:
                    if length > 0:
                        total_count += 1
                        if length > 32000:
                            filtered_count += 1
                        else:
                            all_lengths.append(length)
    
    if filtered_count > 0:
        print(f"Warning: Filtered out {filtered_count} responses longer than 32,000 tokens ({filtered_count/total_count*100:.1f}% of total)")
    
    return all_lengths


def extract_passed_response_lengths(df: pd.DataFrame, threshold: float = 1.0) -> List[int]:
    """Extract response lengths for passed responses only, filtering to 32k limit."""
    passed_lengths = []
    filtered_count = 0
    total_passed_count = 0
    
    for _, row in df.iterrows():
        if ('response_lengths' in row and 'scores' in row and 
            row['response_lengths'] is not None and row['scores'] is not None):
            lengths = row['response_lengths']
            scores = row['scores']
            if isinstance(lengths, (list, np.ndarray)) and isinstance(scores, (list, np.ndarray)):
                for length, score in zip(lengths, scores):
                    if score >= threshold and length > 0:
                        total_passed_count += 1
                        if length > 32000:
                            filtered_count += 1
                        else:
                            passed_lengths.append(length)
    
    if filtered_count > 0:
        print(f"Warning: Filtered out {filtered_count} passed responses longer than 32,000 tokens ({filtered_count/total_passed_count*100:.1f}% of passed responses)")
    
    return passed_lengths


def plot_pass_rate_distribution(pass_rates: List[float], output_dir: Path, model_name: str):
    """Plot pass rate distribution."""
    if not pass_rates:
        print("No pass rate data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Histogram only - cleaner and more readable
    plt.hist(pass_rates, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Pass Rate', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Pass Rate Distribution - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    mean_pass_rate = np.mean(pass_rates)
    median_pass_rate = np.median(pass_rates)
    std_pass_rate = np.std(pass_rates)
    
    stats_text = f'Mean: {mean_pass_rate:.3f}\nMedian: {median_pass_rate:.3f}\nStd: {std_pass_rate:.3f}\nCount: {len(pass_rates)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{model_name}_pass_rate_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pass rate distribution saved to: {output_file}")
    
    # Print statistics
    mean_pass_rate = np.mean(pass_rates)
    median_pass_rate = np.median(pass_rates)
    std_pass_rate = np.std(pass_rates)
    
    print(f"Pass Rate Statistics:")
    print(f"  Mean: {mean_pass_rate:.3f}")
    print(f"  Median: {median_pass_rate:.3f}")
    print(f"  Std: {std_pass_rate:.3f}")
    print(f"  Min: {np.min(pass_rates):.3f}")
    print(f"  Max: {np.max(pass_rates):.3f}")


def plot_length_distribution(all_lengths: List[int], passed_lengths: List[int], 
                           output_dir: Path, model_name: str):
    """Plot response length distributions with improved readability."""
    if not all_lengths:
        print("No response length data to plot")
        return
    
    plt.figure(figsize=(15, 8))
    
    # Log scale histogram comparison
    plt.subplot(2, 2, 1)
    bins = np.logspace(np.log10(min(all_lengths)), np.log10(max(all_lengths)), 40)
    plt.hist(all_lengths, bins=bins, alpha=0.7, label='All Responses', density=True, color='skyblue')
    if passed_lengths:
        plt.hist(passed_lengths, bins=bins, alpha=0.7, label='Passed Responses', density=True, color='lightcoral')
    plt.xlabel('Response Length (tokens)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Response Length Distribution (Log Scale) - {model_name}', fontsize=13, fontweight='bold')
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Linear scale histogram for detailed view
    plt.subplot(2, 2, 2)
    plt.hist(all_lengths, bins=50, alpha=0.7, label='All Responses', density=True, color='skyblue')
    if passed_lengths:
        plt.hist(passed_lengths, bins=50, alpha=0.7, label='Passed Responses', density=True, color='lightcoral')
    plt.xlabel('Response Length (tokens)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Response Length Distribution (Linear Scale) - {model_name}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # CDF comparison
    plt.subplot(2, 2, 3)
    sorted_all = np.sort(all_lengths)
    cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
    plt.plot(sorted_all, cdf_all, label='All Responses', linewidth=2, color='steelblue')
    
    if passed_lengths:
        sorted_passed = np.sort(passed_lengths)
        cdf_passed = np.arange(1, len(sorted_passed) + 1) / len(sorted_passed)
        plt.plot(sorted_passed, cdf_passed, label='Passed Responses', linewidth=2, color='crimson')
    
    plt.xlabel('Response Length (tokens)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title(f'Response Length CDF - {model_name}', fontsize=13, fontweight='bold')
    plt.xscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Statistics comparison
    plt.subplot(2, 2, 4)
    stats_text_all = (f'All Responses:\n'
                      f'Mean: {np.mean(all_lengths):.1f}\n'
                      f'Median: {np.median(all_lengths):.1f}\n'
                      f'Count: {len(all_lengths)}')
    
    if passed_lengths:
        stats_text_passed = (f'Passed Responses:\n'
                            f'Mean: {np.mean(passed_lengths):.1f}\n'
                            f'Median: {np.median(passed_lengths):.1f}\n'
                            f'Count: {len(passed_lengths)}')
        stats_text = f'{stats_text_all}\n\n{stats_text_passed}'
    else:
        stats_text = stats_text_all
    
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
             fontsize=11, verticalalignment='center')
    plt.axis('off')
    plt.title(f'Response Length Statistics - {model_name}', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{model_name}_response_length_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Response length distribution saved to: {output_file}")
    
    # Print statistics
    print(f"Response Length Statistics:")
    print(f"  All Responses - Mean: {np.mean(all_lengths):.1f}, Median: {np.median(all_lengths):.1f}, Count: {len(all_lengths)}")
    if passed_lengths:
        print(f"  Passed Responses - Mean: {np.mean(passed_lengths):.1f}, Median: {np.median(passed_lengths):.1f}, Count: {len(passed_lengths)}")


def plot_pass_rate_vs_length(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Plot pass rate vs average response length correlation."""
    pass_rates = []
    avg_lengths = []
    filtered_entries = 0
    total_entries = 0
    
    for _, row in df.iterrows():
        if ('pass_rate' in row and 'response_lengths' in row and 
            row['pass_rate'] is not None and row['response_lengths'] is not None):
            lengths = row['response_lengths']
            if isinstance(lengths, (list, np.ndarray)) and len(lengths) > 0:
                total_entries += 1
                valid_lengths = [l for l in lengths if 0 < l <= 32000]
                filtered_lengths = [l for l in lengths if l > 32000]
                
                if filtered_lengths:
                    filtered_entries += 1
                
                if valid_lengths:
                    pass_rates.append(row['pass_rate'])
                    avg_lengths.append(np.mean(valid_lengths))
    
    if filtered_entries > 0:
        print(f"Warning: {filtered_entries} entries had responses longer than 32,000 tokens ({filtered_entries/total_entries*100:.1f}% of entries)")
    
    if not pass_rates:
        print("No data for pass rate vs length correlation")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(avg_lengths, pass_rates, alpha=0.6, color='steelblue', s=30)
    plt.xlabel('Average Response Length (tokens)', fontsize=12)
    plt.ylabel('Pass Rate', fontsize=12)
    plt.title(f'Pass Rate vs Average Response Length - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Correlation coefficient
    correlation = np.corrcoef(avg_lengths, pass_rates)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
             fontsize=11)
    
    # Hexbin plot for density
    plt.subplot(1, 2, 2)
    plt.hexbin(avg_lengths, pass_rates, gridsize=25, cmap='Blues', mincnt=1)
    plt.xlabel('Average Response Length (tokens)', fontsize=12)
    plt.ylabel('Pass Rate', fontsize=12)
    plt.title(f'Pass Rate vs Length Density - {model_name}', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(label='Count')
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistics to the density plot
    stats_text = (f'Length Mean: {np.mean(avg_lengths):.1f}\n'
                  f'Pass Rate Mean: {np.mean(pass_rates):.3f}\n'
                  f'Sample Count: {len(pass_rates)}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'{model_name}_pass_rate_vs_length.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pass rate vs length correlation saved to: {output_file}")
    print(f"Correlation between pass rate and average response length: {correlation:.3f}")


def collect_file_statistics(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """Collect statistics for a single file."""
    stats = {"file_name": file_name}
    
    # Extract data
    pass_rates = extract_pass_rates(df)
    response_lengths = extract_response_lengths(df)
    
    # Pass rate statistics
    if pass_rates:
        stats.update({
            "pass_rate_mean": np.mean(pass_rates),
            "pass_rate_median": np.median(pass_rates),
            "pass_rate_std": np.std(pass_rates),
            "pass_rate_min": np.min(pass_rates),
            "pass_rate_max": np.max(pass_rates),
            "pass_rate_count": len(pass_rates)
        })
    else:
        stats.update({
            "pass_rate_mean": 0, "pass_rate_median": 0, "pass_rate_std": 0,
            "pass_rate_min": 0, "pass_rate_max": 0, "pass_rate_count": 0
        })
    
    # Response length statistics
    if response_lengths:
        stats.update({
            "length_min": np.min(response_lengths),
            "length_max": np.max(response_lengths),
            "length_25th": np.percentile(response_lengths, 25),
            "length_75th": np.percentile(response_lengths, 75),
            "length_mean": np.mean(response_lengths),
            "length_median": np.median(response_lengths),
            "length_count": len(response_lengths)
        })
    else:
        stats.update({
            "length_min": 0, "length_max": 0, "length_25th": 0,
            "length_75th": 0, "length_mean": 0, "length_median": 0, "length_count": 0
        })
    
    return stats


def print_summary_table(all_file_stats: List[Dict[str, Any]]):
    """Print a comprehensive summary table of all processed files."""
    if not all_file_stats:
        print("No files processed for summary.")
        return
    
    print("\n" + "="*120)
    print("SUMMARY STATISTICS FOR ALL FILES")
    print("="*120)
    
    # Pass Rate Summary Table
    print("\nPASS RATE STATISTICS:")
    print("-" * 100)
    header = f"{'File Name':<35} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Count':<8}"
    print(header)
    print("-" * 100)
    
    for stats in all_file_stats:
        row = (f"{stats['file_name'][:33]:<35} "
               f"{stats['pass_rate_mean']:<8.3f} "
               f"{stats['pass_rate_median']:<8.3f} "
               f"{stats['pass_rate_std']:<8.3f} "
               f"{stats['pass_rate_min']:<8.3f} "
               f"{stats['pass_rate_max']:<8.3f} "
               f"{stats['pass_rate_count']:<8}")
        print(row)
    
    # Response Length Summary Table
    print("\nRESPONSE LENGTH STATISTICS:")
    print("-" * 110)
    header = f"{'File Name':<35} {'Min':<8} {'25%':<8} {'Mean':<8} {'Median':<8} {'75%':<8} {'Max':<8} {'Count':<8}"
    print(header)
    print("-" * 110)
    
    for stats in all_file_stats:
        row = (f"{stats['file_name'][:33]:<35} "
               f"{stats['length_min']:<8.0f} "
               f"{stats['length_25th']:<8.0f} "
               f"{stats['length_mean']:<8.0f} "
               f"{stats['length_median']:<8.0f} "
               f"{stats['length_75th']:<8.0f} "
               f"{stats['length_max']:<8.0f} "
               f"{stats['length_count']:<8}")
        print(row)
    
    # Overall Summary
    print("\nOVERALL SUMMARY:")
    print("-" * 60)
    
    # Aggregate pass rate statistics
    all_pass_rates = []
    all_lengths = []
    total_files = len(all_file_stats)
    
    for stats in all_file_stats:
        if stats['pass_rate_count'] > 0:
            # Weight by count for proper aggregation
            all_pass_rates.extend([stats['pass_rate_mean']] * stats['pass_rate_count'])
        if stats['length_count'] > 0:
            all_lengths.extend([stats['length_mean']] * stats['length_count'])
    
    if all_pass_rates:
        overall_pass_rate_mean = np.mean(all_pass_rates)
        overall_pass_rate_std = np.std(all_pass_rates)
        print(f"Overall Pass Rate - Mean: {overall_pass_rate_mean:.3f}, Std: {overall_pass_rate_std:.3f}")
    
    if all_lengths:
        overall_length_mean = np.mean(all_lengths)
        overall_length_std = np.std(all_lengths)
        print(f"Overall Length - Mean: {overall_length_mean:.0f}, Std: {overall_length_std:.0f} tokens")
    
    print(f"Total Files Processed: {total_files}")
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Visualize pass rate and response length distributions")
    
    parser.add_argument("input_path", help="Directory or parquet file path with scored data")
    parser.add_argument("--output_dir", default="./figures", help="Output directory for figures")
    parser.add_argument("--model_name", default="model", help="Model name for plot titles")
    parser.add_argument("--pattern", default="*.parquet", help="File pattern for directories")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive search")
    parser.add_argument("--correct_reward_threshold", type=float, default=1.0, help="Pass threshold")
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Input path does not exist: {args.input_path}")
        return
    
    # Find files
    if input_path.is_file():
        parquet_files = [input_path]
    else:
        if args.recursive:
            parquet_files = list(input_path.rglob(args.pattern))
        else:
            parquet_files = list(input_path.glob(args.pattern))
        
        if not parquet_files:
            print(f"No files matching '{args.pattern}' found")
            return
        
        parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Process each file individually
    processed_files = 0
    all_file_stats = []
    
    for file_path in parquet_files:
        print(f"\nProcessing: {file_path.name}")
        df = read_parquet_file(str(file_path))
        
        if df.empty:
            print(f"  Skipping empty file: {file_path.name}")
            continue
        
        # Extract data for this file
        pass_rates = extract_pass_rates(df)
        response_lengths = extract_response_lengths(df)
        passed_lengths = extract_passed_response_lengths(df, args.correct_reward_threshold)
        
        print(f"  Found {len(pass_rates)} pass rates, {len(response_lengths)} response lengths")
        
        if not pass_rates and not response_lengths:
            print(f"  No data found in {file_path.name}")
            continue
        
        # Collect statistics for this file
        file_stats = collect_file_statistics(df, file_path.name)
        all_file_stats.append(file_stats)
        
        # Create file-specific model name
        file_stem = file_path.stem
        file_model_name = f"{args.model_name}_{file_stem}" if args.model_name != "model" else file_stem
        
        # Create file-specific output directory
        file_output_dir = output_dir / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Generating figures for {file_path.name}...")
        
        # Generate visualizations for this file
        if pass_rates:
            plot_pass_rate_distribution(pass_rates, file_output_dir, file_model_name)
        
        if response_lengths:
            plot_length_distribution(response_lengths, passed_lengths, file_output_dir, file_model_name)
        
        if pass_rates and response_lengths:
            plot_pass_rate_vs_length(df, file_output_dir, file_model_name)
        
        processed_files += 1
        print(f"  Figures for {file_path.name} saved to: {file_output_dir}")
    
    print(f"\nProcessed {processed_files}/{len(parquet_files)} files successfully")
    print(f"All figures saved under: {output_dir}")
    
    # Print summary table for all files
    if all_file_stats:
        print_summary_table(all_file_stats)


if __name__ == "__main__":
    main()
