import os
import json
import argparse
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    
    all_metrics = {}
    results_dir = args.results_dir
    for model_result_dir in os.listdir(results_dir):
        # print(model_result_dir)
        if model_result_dir == 'logs' or model_result_dir.endswith('.json'):
            continue
        
        for result_file in sorted(os.listdir(os.path.join(results_dir, model_result_dir))):
            if 'metric' in result_file:
                with open(os.path.join(results_dir, model_result_dir, result_file), 'r') as f:
                    data = json.load(f)
                if model_result_dir not in all_metrics:
                    all_metrics[model_result_dir] = {}
                all_metrics[model_result_dir][result_file.split('.')[0]] = data
                
    # Print the summary
    print(all_metrics)
    rows = []
    for model_result_dir in sorted(all_metrics):
        row = []
        if model_result_dir.startswith("[new]"):
            row.append(model_result_dir)
            for result_file in sorted(all_metrics[model_result_dir], reverse=True):
                # row.append(result_file)
                row.append(all_metrics[model_result_dir][result_file]["pass@1_(avg32)"])
            rows.append(row)
    print(tabulate(rows, headers=["model", "aime24", "aime25"], tablefmt="tsv", numalign="right", floatfmt=".2f"))
        
    # Save as a `summary.json` file
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    main()
