#!/bin/bash
#SBATCH --job-name=example-eval-guru-7b
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%j_%x.log
#SBATCH --error=slurm/%j_%x.log
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=main


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
CONDA_BIN_PATH=/mnt/weka/home/yuqi.wang/miniconda3/envs/Reasoning360/bin

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
# export GLOO_SOCKET_IFNAME=ens10f0np0

unset LD_LIBRARY_PATH

# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 $CONDA_BIN_PATH/ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    $CONDA_BIN_PATH/ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        $CONDA_BIN_PATH/ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10


# =================== (Offline) Leaderboard Eval Config ===================
leaderboard_list=(
    # MATH
    "aime"
    "aime2025"
    # "math"
    # CODE GENERATION
    # "mbpp"
    # "humaneval"
    # "livecodebench"
    # LOGIC
    # "arcagi1"
    # "zebra_puzzle_dataset"
    # STEM
    # "gpqa_diamond"
    # "supergpqa"
    # TABLE
    # "finqa"
    # "hitab"
    # "multihier"
    # SIMULATION
    # "codeio"
    # "cruxeval-i"
    # "cruxeval-o"
    # OTHERS
#     "livebench_reasoning"
#     "livebench_language"
#     "livebench_data_analysis"
#     "ifeval"
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

declare -A exp_mappings

exp_mappings["/mnt/sharefs/users/shibo.hao/llama3.1-70B-yz/saves/Llama-3.1-70B-AM-Thinking-v1-old"]="2025-06-10-amthink-sft-llama70b-am-think-v1"
exp_mappings["/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-old"]="2025-06-10-amthink-sft-qwen32b-am-think-v1"
exp_mappings["/mnt/sharefs/users/shibo.hao/llama3.1-70B-yz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-old"]="2025-06-10-amthink-sft-qwen32b-am-think-v1-setting2"
exp_mappings["/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-R1-0528"]="2025-06-10-amthink-sft-qwen32b-r1-0528"
exp_mappings["/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-OpenThoughts3-1.2M"]="2025-06-18-openthoughts3-sft-qwen32b"
exp_mappings["/mnt/sharefs/users/shibo.hao/llama3.1-70b-yz/saves/Llama-3.1-70B-OpenThoughts3-1.2M-5epochs-lr5e-5"]="2025-06-18-openthoughts3-sft-llama70b"


n_nodes=$SLURM_NNODES
n_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7

SHARED_DATA_PATH=/mnt/sharefs/users/zhuojun.cheng
data_folder=${SHARED_DATA_PATH}/guru_data/test/offline_leaderboard_release_0603/
save_folder=./evaluation_results/test_offline_leaderboard_output/
# model_path=/mnt/sharefs/users/yuqi.wang/other_ckpts/DeepSeek-R1-Distill-Llama-70B
# model_path=/mnt/sharefs/users/yuqi.wang/other_ckpts/Qwen3-32B
# model_path=/mnt/sharefs/users/yuqi.wang/other_ckpts/Qwen2.5-32B-Instruct
# model_path=/mnt/sharefs/users/haonan.li/Qwen2.5-32B-instruct-think_pattern_base/checkpoint-222
# model_path=/mnt/sharefs/users/haonan.li/Qwen2.5-32B-instruct-think_pattern_simple
# model_path=/mnt/sharefs/users/haonan.li/Qwen2.5-32B-instruct-think_pattern_complex
# model_path=/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-OpenThoughts3-1.2M/checkpoint-2900
# model_path=/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-old/checkpoint-550
# model_path=/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-old/checkpoint-1084
# model_path=/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-R1-0528/checkpoint-700
model_path=/mnt/sharefs/users/shibo.hao/tz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-R1-0528/checkpoint-1422
# model_path=/mnt/sharefs/users/shibo.hao/llama3.1-70B-yz/saves/Qwen2.5-32B-base-AM-thinking-distilled-v1-old/checkpoint-2168

# Extract model name from the path
model_path_basename=$(basename "$model_path")
if [[ $model_path_basename == "checkpoint-"* ]]; then
  model_path_dirname=$(dirname "$model_path")
  model_name=${exp_mappings[$model_path_dirname]}-$model_path_basename
else
  model_name=$model_path_basename
fi
echo $model_name

# Check if leaderboard generation folder exists, create if it doesn't
if [ ! -d "$save_folder" ]; then
    mkdir -p "$save_folder"
    echo "Leaderboard output path created: ${save_folder}"
else
    echo "Leaderboard output path ${save_folder} already exists"
fi

# Create a logs directory inside save_folder if it doesn't exist
logs_dir="${save_folder}logs/"
if [ ! -d "$logs_dir" ]; then
    mkdir -p "$logs_dir"
    echo "Logs directory created: ${logs_dir}"
fi

# =================== Generation Config ===================
# Sampling:
#  If samples <= 400, do sampling, sample at least 4x for each sample
#  If samples >1000, random sample 1000

# Length limits:
#  If `multihier` or `arcagi1`, use 16k prompt length, 16k response length
#  Otherwise, use 4k prompt length, 28k response length

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
        n_samples=4
    else
        n_samples=1
    fi

    batch_size=1024
    temperature=0.6
    top_p=0.95
    top_k=-1 # 0 for hf rollout, -1 for vllm rollout

    if [ "$leaderboard" == "arcagi1" ]; then
        prompt_length=16384
        response_length=16384
    else
        prompt_length=4096
        response_length=28672
    fi
    tensor_model_parallel_size=8
    gpu_memory_utilization=0.7
    
    # Create log files - one for generation and one for evaluation
    gen_log_file="${logs_dir}${model_name}_${leaderboard}_gen.log"
    eval_log_file="${logs_dir}${model_name}_${leaderboard}_eval.log"
    
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
    
    echo "Processing $leaderboard: $data_file -> $save_path" | tee -a "$gen_log_file"
    
    export CUDA_VISIBLE_DEVICES=${gpu_ids}

    # Generation step with tee to generation log file
    echo "Starting generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
    {
        ${CONDA_BIN_PATH}/python -m verl.trainer.main_generation \
            trainer.nnodes=$n_nodes \
            trainer.n_gpus_per_node=$n_gpus_per_node \
            data.path="$data_file" \
            data.prompt_key=prompt \
            data.n_samples=$n_samples \
            data.batch_size=$batch_size \
            data.output_path="$save_path" \
            model.path=$model_path \
            +model.trust_remote_code=True \
            rollout.temperature=$temperature \
            rollout.top_k=$top_k \
            rollout.top_p=$top_p \
            rollout.prompt_length=$prompt_length \
            rollout.response_length=$response_length \
            rollout.max_num_batched_tokens=$(($prompt_length + $response_length)) \
            rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
            rollout.gpu_memory_utilization=$gpu_memory_utilization
    } 2>&1 | tee -a "$gen_log_file"
    echo "Completed generation for $leaderboard at $(date)" | tee -a "$gen_log_file"

    # Evaluation step with tee to evaluation log file
    echo "Starting evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    unset LD_LIBRARY_PATH
    {
        ${CONDA_BIN_PATH}/python -m verl.trainer.main_eval \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model # this indicates key "reference" in the reward model data is the ground truth
    } 2>&1 | tee -a "$eval_log_file"
    echo "Completed evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"

    echo "Completed processing $leaderboard. Generation log: $gen_log_file, Evaluation log: $eval_log_file"
done