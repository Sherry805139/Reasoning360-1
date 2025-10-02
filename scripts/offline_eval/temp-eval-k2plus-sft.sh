#!/bin/bash
#SBATCH --job-name=temp-param-study
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm/param_study_%A_%a.log
#SBATCH --error=slurm/param_study_%A_%a.log
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --array=4  # Adjust this range based on number of temperature values

# Define temperature values for the parametric study
TEMPERATURES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)

# Get the temperature value for this array job
temperature=${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}

echo "Starting parametric study with temperature: $temperature"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Create temperature ID for folder naming (use Python regex to replace '.' with '_')
temp_for_folder=$(python3 -c 'import re,sys; print(re.sub(r"\\.", "_", sys.argv[1]))' "$temperature")

# =================== Frequently Used Variables ===================
export STEM_LLM_JUDGE_URL="http://azure-uk-hpc-H200-instance-013:8000"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

# =================== Cluster Environment ===================
export NCCL_DEBUG=info
export NCCL_ALGO=NVLSTree
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

# We'll track PIDs so we can wait on them and detect errors
declare -A pids

# Spawn each check in the background
for host in "${nodes[@]}"; do
    echo "Spawning GPU check on node: $host"
    srun --nodes=1 --ntasks=1 --nodelist="$host" \
         ~/Reasoning360/scripts/tools/check_gpu.sh &
    pids["$host"]=$!
done

# Now wait for each job to finish and capture errors
error_found=0
for host in "${nodes[@]}"; do
    # wait returns the exit code of the process
    if ! wait "${pids[$host]}"; then
        echo "ERROR: Found GPU usage by other users on $host. Exiting."
        error_found=1
    fi
done

if [[ $error_found -eq 1 ]]; then
    exit 1
fi

echo "=== No leftover GPU usage found on all allocated nodes. ==="
echo "Proceeding with the main job..."

export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
# export GLOO_SOCKET_IFNAME=ens10f0np0

unset LD_LIBRARY_PATH

# =================== Ray start ===================
ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ${CONDA_BIN_PATH}ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10

# =================== (Offline) Leaderboard Eval Config ===================
leaderboard_list=(
    # MATH
    "aime"
    # "aime2025"
    "math"
    # CODE GENERATION
    "mbpp"
    # "humaneval"
    # "livecodebench"
    # LOGIC
    # "arcagi1"
    "zebra_puzzle_dataset"
    # STEM
    "gpqa_diamond"
    # "supergpqa"
    # TABLE
    # "finqa"
    "hitab"
    # "multihier"
    # SIMULATION
    # "codeio"
    # "cruxeval-i"
    # "cruxeval-o"
    # OTHERS
    # "livebench_reasoning"
    # "livebench_language"
    # "livebench_data_analysis"
    # "ifeval"
)

# Define domain mappings for each leaderboard
declare -A domain_mappings

domain_mappings["aime"]="math"
domain_mappings["aime2025"]="math"
domain_mappings["math"]="math"
domain_mappings["humaneval"]="codegen"
domain_mappings["livecodebench"]="codegen"
domain_mappings["mbpp"]="codegen"
domain_mappings["arcagi1"]="logic"
domain_mappings["zebra_puzzle_dataset"]="logic"
domain_mappings["finqa"]="table"
domain_mappings["hitab"]="table"
domain_mappings["multihier"]="table"
domain_mappings["codeio"]="simulation"
domain_mappings["cruxeval-i"]="simulation"
domain_mappings["cruxeval-o"]="simulation"
domain_mappings["gpqa_diamond"]="stem"
domain_mappings["supergpqa"]="stem"
domain_mappings["livebench_reasoning"]="ood"
domain_mappings["livebench_language"]="ood"
domain_mappings["livebench_data_analysis"]="ood"
domain_mappings["ifeval"]="ood"

n_nodes=$SLURM_NNODES
n_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7

SHARED_DATA_PATH=/lustrefs/users/zhuojun.cheng/vpim/guru_data
SHARED_MODEL_PATH=/lustrefs/users/runner/workspace/checkpoints/huggingface/sft
data_folder=${SHARED_DATA_PATH}/test/offline_leaderboard_release_0603
save_folder=./evaluation_results/lng131k/am_offline_output_temp_${temp_for_folder}
model_path=${SHARED_MODEL_PATH}/mid4_sft_reasoning_am_cos_epoch/checkpoints/checkpoint_0002250
# model_path=${SHARED_MODEL_PATH}/mid4_sft_reasoning_am/checkpoints/checkpoint_0001500
# model_path=${SHARED_MODEL_PATH}/mid4_sft_reasoning_ot/checkpoints/checkpoint_0001500

# Extract model name from the path
model_name=$(basename "$model_path")

# Check if leaderboard generation folder exists, create if it doesn't
if [ ! -d "$save_folder" ]; then
    mkdir -p "$save_folder"
    echo "Leaderboard output path created: ${save_folder}"
else
    echo "Leaderboard output path ${save_folder} already exists"
fi

# Create a logs directory inside save_folder if it doesn't exist
logs_dir="${save_folder}/logs"
if [ ! -d "$logs_dir" ]; then
    mkdir -p "$logs_dir"
    echo "Logs directory created: ${logs_dir}"
fi

echo "Processing temperature: $temperature"
echo "Save folder: $save_folder"

# =================== Generation Config ===================
# Sampling:
#  If samples <= 400, do sampling, sample at least 4x for each sample
#     TWK NOTE: I've adjusted this to be at least 16x (we do 32x for AIME)
#  If samples >1000, random sample 1000
#     TWK NOTE: I've also increased n_samples here to be 16x to ensure that capture sufficient variability

# Length limits:
#  If `multihier` or `arcagi1`, use 16k prompt length, 16k response length (TWK: ~114k)
#  Otherwise, use 4k prompt length, 28k response length (TWK: 126k )
#    TWK NOTE: I've expanded this to take advantage of K2+'s native ~132k context length

# Batch size:
#  Always use 1024

# Generation parameters:
#  Always use temperature=1.0, top_p=0.7

for leaderboard in "${leaderboard_list[@]}"; do
    # Get the domain for this leaderboard
    domain=${domain_mappings[$leaderboard]}
    
    # Sampling
    if [ "$leaderboard" == "aime" ] || [ "$leaderboard" == "aime2025" ]; then
        # Note our test set is repeated 8x, so it's sampling 32x
        n_samples=4  
    elif [ "$leaderboard" == "arcagi1" ] || [ "$leaderboard" == "livecodebench" ] || [ "$leaderboard" == "humaneval" ] || [ "$leaderboard" == "zebra_puzzle_dataset" ] || [ "$leaderboard" == "multihier" ] || [ "$leaderboard" == "codeio" ] || [ "$leaderboard" == "gpqa_diamond" ]; then
        n_samples=16
    else
        n_samples=16
    fi

    batch_size=1024
    top_p=0.7
    top_k=-1 # 0 for hf rollout, -1 for vllm rollout

    if [ "$leaderboard" == "arcagi1" ]; then
        prompt_length=16384
        response_length=16384
    else
        prompt_length=4096
        response_length=28672
    fi
    tensor_model_parallel_size=4
    gpu_memory_utilization=0.7
    
    # Create log files - one for generation and one for evaluation
    gen_log_file="${logs_dir}/${model_name}_${leaderboard}_temp_${temp_for_folder}_gen.log"
    eval_log_file="${logs_dir}/${model_name}_${leaderboard}_temp_${temp_for_folder}_eval.log"
    
    # Find the matching file in the data folder
    if [ "$leaderboard" == "aime" ] || [ "$leaderboard" == "aime2025" ]; then
        file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
    else
        file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
    fi
    
    # Use find to get the actual file path
    data_file=$(find "$data_folder" -name "$file_pattern" -type f | head -n 1)
    echo "data_file: $data_file"

    if [ -z "$data_file" ]; then
        echo "No file found matching pattern: $file_pattern. Skipping." | tee -a "$gen_log_file"
        continue
    fi
    
    # Extract the file name without path
    file_name=$(basename "$data_file")
    save_path="${save_folder}/${model_name}/${file_name}"
    
    echo "Processing $leaderboard with temperature $temperature: $data_file -> $save_path" | tee -a "$gen_log_file"
    
    export CUDA_VISIBLE_DEVICES=${gpu_ids}

    # Generation step with tee to generation log file
    echo "Starting generation for $leaderboard with temperature $temperature at $(date)" | tee -a "$gen_log_file"
    {
        ${CONDA_BIN_PATH}python -m verl.trainer.main_generation \
            trainer.nnodes=$n_nodes \
            trainer.n_gpus_per_node=$n_gpus_per_node \
            data.path="$data_file" \
            data.prompt_key=prompt \
            data.n_samples=$n_samples \
            data.batch_size=$batch_size \
            data.output_path="$save_path" \
            model.path=$model_path \
            +model.trust_remote_code=True \
            +actor_rollout_ref.model.override_config.rope_scaling.type=yarn \
            +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
            +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
            rollout.temperature=$temperature \
            rollout.top_k=$top_k \
            rollout.top_p=$top_p \
            rollout.prompt_length=$prompt_length \
            rollout.response_length=$response_length \
            rollout.max_num_batched_tokens=$(($prompt_length + $response_length)) \
            rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
            rollout.gpu_memory_utilization=$gpu_memory_utilization
    } 2>&1 | tee -a "$gen_log_file"
    echo "Completed generation for $leaderboard with temperature $temperature at $(date)" | tee -a "$gen_log_file"

    # Evaluation step with tee to evaluation log file
    echo "Starting evaluation for $leaderboard with temperature $temperature at $(date)" | tee -a "$eval_log_file"
    unset LD_LIBRARY_PATH
    if [ -z "$save_path" ]; then
        echo "No file found matching: $save_path. Skipping." | tee -a "$eval_log_file"
        continue
    fi
    
    # Set environment variable for temperature tracking in eval script
    export PARAM_STUDY_TEMP=$temperature
    
    {
        ${CONDA_BIN_PATH}python -m verl.trainer.main_eval \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model # this indicates key "reference" in the reward model data is the ground truth
    } 2>&1 | tee -a "$eval_log_file"
    echo "Completed evaluation for $leaderboard with temperature $temperature at $(date)" | tee -a "$eval_log_file"

    echo "Completed processing $leaderboard with temperature $temperature. Generation log: $gen_log_file, Evaluation log: $eval_log_file"
done

# =================== Results Aggregation ===================
echo "Aggregating results for temperature: $temperature"

# Create results summary directory
results_summary_dir="./evaluation_results/temperature_study_summary"
mkdir -p "$results_summary_dir"

# Create a results file for this temperature
temp_results_file="${results_summary_dir}/results_temp_${temp_for_folder}.json"
summary_csv="${results_summary_dir}/temperature_study_summary.csv"

# Initialize results structure for this temperature
cat > "$temp_results_file" << EOF
{
  "temperature": $temperature,
  "model_path": "$model_path",
  "model_name": "$model_name",
  "job_id": "$SLURM_JOB_ID",
  "array_task_id": "$SLURM_ARRAY_TASK_ID",
  "timestamp": "$(date -Iseconds)",
  "results": {}
}
EOF

# Extract results from each leaderboard's evaluation output
python3 << 'PYTHON_EOF'
import json
import os
import glob
import re
import sys
from pathlib import Path

# Read the results file we just created
temp_results_file = os.environ['temp_results_file']
with open(temp_results_file, 'r') as f:
    results = json.load(f)

# Look for evaluation results in the save folder
save_folder = os.environ['save_folder']
model_name = os.environ['model_name']
temperature = float(os.environ['temperature'])

print(f"Looking for results in: {save_folder}/{model_name}")

# Find all result files
result_files = glob.glob(f"{save_folder}/{model_name}/*_results.json")
if not result_files:
    # Try alternative patterns
    result_files = glob.glob(f"{save_folder}/{model_name}/*.json")
    result_files = [f for f in result_files if 'results' in f or 'eval' in f]

print(f"Found {len(result_files)} result files")

for result_file in result_files:
    try:
        # Extract leaderboard name from filename
        filename = os.path.basename(result_file)
        # Try to extract leaderboard name from various filename patterns
        leaderboard_match = re.search(r'(?:math|codegen|logic|table|simulation|stem|ood)__([^_]+)', filename)
        if leaderboard_match:
            leaderboard = leaderboard_match.group(1)
        else:
            # Fallback: use filename without extension
            leaderboard = os.path.splitext(filename)[0]
        
        print(f"Processing {leaderboard} results from {result_file}")
        
        with open(result_file, 'r') as f:
            eval_results = json.load(f)
        
        # Extract key metrics - adapt based on your actual result structure
        metrics = {}
        if isinstance(eval_results, dict):
            # Common metric names to look for
            metric_keys = ['accuracy', 'score', 'pass_rate', 'success_rate', 'avg_score', 'mean_score']
            for key in eval_results:
                if any(metric in key.lower() for metric in metric_keys):
                    metrics[key] = eval_results[key]
                elif key in ['total_samples', 'num_correct', 'num_total']:
                    metrics[key] = eval_results[key]
        
        # If we couldn't find standard metrics, store the whole result
        if not metrics and eval_results:
            metrics = eval_results
        
        results['results'][leaderboard] = metrics
        
    except Exception as e:
        print(f"Error processing {result_file}: {e}")
        results['results'][leaderboard] = {"error": str(e)}

# Write updated results
with open(temp_results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results aggregated for temperature {temperature}")
PYTHON_EOF

# Add this temperature's results to the summary CSV
python3 << 'PYTHON_EOF'
import json
import csv
import os
from pathlib import Path

temp_results_file = os.environ['temp_results_file']
summary_csv = os.environ['summary_csv']

# Load this temperature's results
with open(temp_results_file, 'r') as f:
    results = json.load(f)

temperature = results['temperature']
timestamp = results['timestamp']
job_info = f"{results['job_id']}_{results['array_task_id']}"

# Prepare CSV row
csv_row = {
    'temperature': temperature,
    'timestamp': timestamp,
    'job_info': job_info,
    'model_name': results['model_name']
}

# Add metrics for each leaderboard
for leaderboard, metrics in results['results'].items():
    if isinstance(metrics, dict) and 'error' not in metrics:
        for metric_name, metric_value in metrics.items():
            col_name = f"{leaderboard}_{metric_name}"
            csv_row[col_name] = metric_value
    else:
        csv_row[f"{leaderboard}_status"] = "error" if 'error' in str(metrics) else "completed"

# Write/append to CSV
file_exists = os.path.exists(summary_csv)
with open(summary_csv, 'a' if file_exists else 'w', newline='') as f:
    if csv_row:  # Only proceed if we have data
        writer = csv.DictWriter(f, fieldnames=sorted(csv_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

print(f"Added temperature {temperature} results to summary CSV")
PYTHON_EOF

echo "Results aggregation completed for temperature: $temperature"
echo "Individual results: $temp_results_file"
echo "Summary CSV: $summary_csv"

# =================== Final Summary Generation (only for last array task) ===================
# Create a final comprehensive summary when all jobs are done
python3 << 'PYTHON_EOF'
import json
import os
import glob
from pathlib import Path

results_summary_dir = os.environ['results_summary_dir']
current_temp = float(os.environ['temperature'])
temperatures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]  # Match the array

# Check if this is the last temperature (highest value)
if current_temp == max(temperatures):
    print("Generating final comprehensive summary...")
    
    # Collect all individual temperature results
    all_results = []
    result_files = glob.glob(f"{results_summary_dir}/results_temp_*.json")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            all_results.append(results)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Sort by temperature
    all_results.sort(key=lambda x: x['temperature'])
    
    # Create comprehensive summary
    final_summary = {
        "study_type": "temperature_parameter_study",
        "total_temperatures": len(all_results),
        "temperature_range": [min(r['temperature'] for r in all_results), 
                            max(r['temperature'] for r in all_results)],
        "model_info": {
            "model_path": all_results[0]['model_path'] if all_results else "",
            "model_name": all_results[0]['model_name'] if all_results else ""
        },
        "leaderboards_tested": list(set().union(*[list(r['results'].keys()) for r in all_results])),
        "results_by_temperature": {str(r['temperature']): r for r in all_results}
    }
    
    # Write final summary
    final_summary_file = f"{results_summary_dir}/final_temperature_study_summary.json"
    with open(final_summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"Final comprehensive summary written to: {final_summary_file}")
    print(f"CSV summary available at: {os.environ['summary_csv']}")
    
    # Generate a simple performance comparison
    comparison_file = f"{results_summary_dir}/temperature_performance_comparison.txt"
    with open(comparison_file, 'w') as f:
        f.write("TEMPERATURE PARAMETER STUDY PERFORMANCE COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        for leaderboard in final_summary['leaderboards_tested']:
            f.write(f"Leaderboard: {leaderboard}\n")
            f.write("-" * 30 + "\n")
            
            for temp_str, temp_data in final_summary['results_by_temperature'].items():
                if leaderboard in temp_data['results']:
                    metrics = temp_data['results'][leaderboard]
                    f.write(f"Temperature {temp_str}: {metrics}\n")
            f.write("\n")
    
    print(f"Performance comparison written to: {comparison_file}")
    print("Temperature parameter study aggregation complete!")

PYTHON_EOF

echo "Parametric study completed for temperature: $temperature"