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
#SBATCH --array=4

# Define temperature values for the parametric study
TEMPERATURES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)

# Get the temperature value for this array job
temperature=${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}

echo "Starting parametric study with temperature: $temperature"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Create temperature ID for folder naming
temp_for_folder=$(echo $temperature | sed 's/\./_/')

# =================== Frequently Used Variables ===================
export STEM_LLM_JUDGE_URL="http://azure-uk-hpc-H200-instance-013:8000"

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

declare -A pids

for host in "${nodes[@]}"; do
    echo "Spawning GPU check on node: $host"
    srun --nodes=1 --ntasks=1 --nodelist="$host" \
         ~/Reasoning360/scripts/tools/check_gpu.sh &
    pids["$host"]=$!
done

error_found=0
for host in "${nodes[@]}"; do
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

export CUDA_LAUNCH_BLOCKING=1

unset LD_LIBRARY_PATH

# =================== Ray start ===================
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ${CONDA_BIN_PATH}ray stop

sleep 10
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10

# =================== Dataset Configuration ===================
# Define all datasets to process
declare -A dataset_configs

# Format: dataset_configs["leaderboard_name"]="domain|n_samples|prompt_length|response_length|batch_size|tensor_parallel"

# MATH
dataset_configs["aime"]="math|4|4096|28672|1024|4"
dataset_configs["math"]="math|16|4096|28672|1024|4"

# CODE GENERATION
dataset_configs["mbpp"]="codegen|16|4096|28672|1024|4"
dataset_configs["humaneval"]="codegen|16|4096|28672|1024|4"

# LOGIC
dataset_configs["arcagi1"]="logic|16|16384|16384|1024|4"
dataset_configs["zebra_puzzle_dataset"]="logic|16|4096|28672|1024|4"

# STEM
dataset_configs["gpqa_diamond"]="stem|16|4096|28672|1024|4"

# TABLE
dataset_configs["hitab"]="table|16|4096|28672|1024|4"

# Select which datasets to run (comment out ones you don't want)
# leaderboard_list=(
#     "aime"
#     "math"
#     "mbpp"
#     "zebra_puzzle_dataset"
#     "gpqa_diamond"
#     "hitab"
# )
leaderboard_list=(
    "math"
)

# =================== Model and Path Configuration ===================
SHARED_DATA_PATH=/lustrefs/users/zhuojun.cheng/vpim/guru_data
SHARED_MODEL_PATH=/lustrefs/users/runner/workspace/checkpoints/huggingface/sft
data_folder=${SHARED_DATA_PATH}/test/offline_leaderboard_release_0603
save_folder=./evaluation_results/lng131k_2/am_offline_output_temp_${temp_for_folder}
model_path=${SHARED_MODEL_PATH}/mid4_sft_reasoning_am_cos_epoch/checkpoints/checkpoint_0002250

model_name=$(basename "$model_path")

if [ ! -d "$save_folder" ]; then
    mkdir -p "$save_folder"
    echo "Leaderboard output path created: ${save_folder}"
else
    echo "Leaderboard output path ${save_folder} already exists"
fi

logs_dir="${save_folder}/logs"
if [ ! -d "$logs_dir" ]; then
    mkdir -p "$logs_dir"
    echo "Logs directory created: ${logs_dir}"
fi

# =================== Build Dataset List for Generation ===================
dataset_paths=()
dataset_names=()
dataset_save_paths=()

for leaderboard in "${leaderboard_list[@]}"; do
    # Parse configuration
    IFS='|' read -r domain n_samples prompt_length response_length batch_size tensor_parallel <<< "${dataset_configs[$leaderboard]}"
    
    # Find the matching file in the data folder
    if [ "$leaderboard" == "aime" ] || [ "$leaderboard" == "aime2025" ]; then
        file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
    else
        file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
    fi
    
    data_file=$(find "$data_folder" -name "$file_pattern" -type f | head -n 1)
    
    if [ -z "$data_file" ]; then
        echo "No file found matching pattern: $file_pattern. Skipping."
        continue
    fi
    
    file_name=$(basename "$data_file")
    save_path="${save_folder}/${model_name}/${file_name}"
    
    # Add to arrays
    dataset_paths+=("$data_file")
    dataset_names+=("$leaderboard")
    dataset_save_paths+=("$save_path")
    
    echo "Added dataset: $leaderboard -> $data_file"
done

# =================== Generation Phase (Single Model Load) ===================
echo "Starting generation phase with temperature: $temperature"
echo "Processing ${#dataset_paths[@]} datasets in single run"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PARAM_STUDY_TEMP=$temperature

gen_log_file="${logs_dir}/${model_name}_all_datasets_temp_${temp_for_folder}_gen.log"

# Convert arrays to comma-separated strings for passing to Python
dataset_paths_str=$(IFS=','; echo "${dataset_paths[*]}")
dataset_names_str=$(IFS=','; echo "${dataset_names[*]}")
dataset_save_paths_str=$(IFS=','; echo "${dataset_save_paths[*]}")

# Create a JSON config file for dataset-specific parameters
dataset_config_file="${save_folder}/dataset_configs_temp_${temp_for_folder}.json"
cat > "$dataset_config_file" << EOF
{
EOF

first=true
for leaderboard in "${leaderboard_list[@]}"; do
    if [ -z "${dataset_configs[$leaderboard]}" ]; then
        continue
    fi
    
    IFS='|' read -r domain n_samples prompt_length response_length batch_size tensor_parallel <<< "${dataset_configs[$leaderboard]}"
    
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$dataset_config_file"
    fi
    
    cat >> "$dataset_config_file" << EOF
  "$leaderboard": {
    "domain": "$domain",
    "n_samples": $n_samples,
    "prompt_length": $prompt_length,
    "response_length": $response_length,
    "batch_size": $batch_size,
    "tensor_model_parallel_size": $tensor_parallel
  }
EOF
done

cat >> "$dataset_config_file" << EOF

}
EOF

echo "Dataset configuration written to: $dataset_config_file"

# Build Hydra list format for paths
hydra_paths="["
hydra_names="["
hydra_outputs="["

for i in "${!dataset_paths[@]}"; do
    if [ $i -gt 0 ]; then
        hydra_paths+=","
        hydra_names+=","
        hydra_outputs+=","
    fi
    hydra_paths+="${dataset_paths[$i]}"
    hydra_names+="${dataset_names[$i]}"
    hydra_outputs+="${dataset_save_paths[$i]}"
done

hydra_paths+="]"
hydra_names+="]"
hydra_outputs+="]"

echo "Hydra paths format: $hydra_paths"
echo "Hydra names format: $hydra_names"
echo "Hydra outputs format: $hydra_outputs"

# Run generation for all datasets
echo "Starting multi-dataset generation at $(date)" | tee -a "$gen_log_file"
{
    ${CONDA_BIN_PATH}python -m verl.trainer.main_generation_multi_dset \
        trainer.nnodes=$SLURM_NNODES \
        trainer.n_gpus_per_node=8 \
        "+data.paths=$hydra_paths" \
        "+data.dataset_names=$hydra_names" \
        "+data.output_paths=$hydra_outputs" \
        +data.dataset_config_file="$dataset_config_file" \
        data.prompt_key=prompt \
        model.path=$model_path \
        +model.trust_remote_code=True \
        +actor_rollout_ref.model.override_config.rope_scaling.type=yarn \
        +actor_rollout_ref.model.override_config.rope_scaling.factor=4.0 \
        +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
        rollout.temperature=$temperature \
        rollout.top_k=-1 \
        rollout.top_p=0.7 \
        rollout.gpu_memory_utilization=0.7
} 2>&1 | tee -a "$gen_log_file"
echo "Completed generation for all datasets with temperature $temperature at $(date)" | tee -a "$gen_log_file"

# =================== Evaluation Phase ===================
echo "Starting evaluation phase"

for i in "${!dataset_save_paths[@]}"; do
    save_path="${dataset_save_paths[$i]}"
    leaderboard="${dataset_names[$i]}"
    
    eval_log_file="${logs_dir}/${model_name}_${leaderboard}_temp_${temp_for_folder}_eval.log"
    
    echo "Starting evaluation for $leaderboard with temperature $temperature at $(date)" | tee -a "$eval_log_file"
    
    {
        ${CONDA_BIN_PATH}python -m verl.trainer.main_eval_multi_dset \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model
    } 2>&1 | tee -a "$eval_log_file"
    
    echo "Completed evaluation for $leaderboard with temperature $temperature at $(date)" | tee -a "$eval_log_file"
done

# =================== Results Aggregation ===================
echo "Aggregating results for temperature: $temperature"

results_summary_dir="./evaluation_results/temperature_study_summary"
mkdir -p "$results_summary_dir"

temp_results_file="${results_summary_dir}/results_temp_${temp_for_folder}.json"
summary_csv="${results_summary_dir}/temperature_study_summary.csv"

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

# Extract and aggregate results
python3 << 'PYTHON_EOF'
import json
import os
import glob
import re

temp_results_file = os.environ['temp_results_file']
with open(temp_results_file, 'r') as f:
    results = json.load(f)

save_folder = os.environ['save_folder']
model_name = os.environ['model_name']
temperature = float(os.environ['temperature'])

print(f"Looking for results in: {save_folder}/{model_name}")

result_files = glob.glob(f"{save_folder}/{model_name}/*_eval_results.json")
print(f"Found {len(result_files)} result files")

for result_file in result_files:
    try:
        filename = os.path.basename(result_file)
        leaderboard_match = re.search(r'(?:math|codegen|logic|table|simulation|stem|ood)__([^_]+)', filename)
        if leaderboard_match:
            leaderboard = leaderboard_match.group(1)
        else:
            leaderboard = os.path.splitext(filename)[0].replace('_eval_results', '')
        
        print(f"Processing {leaderboard} results from {result_file}")
        
        with open(result_file, 'r') as f:
            eval_results = json.load(f)
        
        if 'detailed_metrics' in eval_results:
            results['results'][leaderboard] = eval_results['detailed_metrics']
        elif 'summary_metrics' in eval_results:
            results['results'][leaderboard] = eval_results['summary_metrics']
        else:
            results['results'][leaderboard] = eval_results
        
    except Exception as e:
        print(f"Error processing {result_file}: {e}")
        results['results'][leaderboard] = {"error": str(e)}

with open(temp_results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results aggregated for temperature {temperature}")
PYTHON_EOF

echo "Results aggregation completed for temperature: $temperature"
echo "Parametric study completed for temperature: $temperature"