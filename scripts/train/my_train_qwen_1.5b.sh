#!/bin/bash
#SBATCH --job-name=example-multinode-rl-qwen32b-base
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --exclusive
#SBATCH --time=720:00:00
SLURM_NNODES=1
SLURM_CPUS_PER_TASK=96
NUM_GPUS=4

# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch
export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

#export NETRC="piaohongming02-city-university-of-hong-kong"
#export WANDB_API_KEY="3b7a3bc7f4c002c90914d317e41221b5d41137b6"

# =================== Cluster Environment ===================
#export NCCL_DEBUG=info
#export NCCL_ALGO=NVLSTree
#export NCCL_IBEXT_DISABLE=1
#export NCCL_NVLS_ENABLE=1
#export NCCL_IB_HCA=mlx5
#export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_P2P_DISABLE=1 

# Get the list of allocated nodes
gpu_ids=4,5,6,7
export CUDA_VISIBLE_DEVICES=${gpu_ids} 

nodes=("127.0.0.1")
echo "Nodes to check: ${nodes[@]}"

# We'll track PIDs so we can wait on them and detect errors
declare -A pids
export head_node=${nodes[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip="127.0.0.1"
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# Redirect Ray temporary directory to a larger disk to avoid /tmp exhaustion
export RAY_TMPDIR=/data1/ray_tmp
mkdir -p "$RAY_TMPDIR"

# Choose a larger local directory for checkpoints to avoid write failures
CHECKPOINT_DIR=/home/hmpiao/hmpiao/xuerong
mkdir -p "$CHECKPOINT_DIR"

# =================== Data Mixture ===================
SHARED_DATA_PATH=/home/hmpiao/adv_reason/Reasoning360/data
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/
TEST_DATA_DIR=${SHARED_DATA_PATH}/online_eval/

# Math (train)
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_440.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__taco_8.8k.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet
# Logic (test)
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet
arcagi_test_path=${TEST_DATA_DIR}/logic__arcagi1_200.parquet

# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_200.parquet

# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_200.parquet

# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet
# Stem (test)
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet

train_files="['${math_train_path}']"  # Use math as example, add to more tasks as needed
test_files="['${math_test_path}','${aime_test_path}']"  # Use math as example, add to more tasks as needed


# =================== Model ===================
BASE_MODEL=/home/hmpiao/hmpiao/Qwen3-1.7B-Base

# =================== Logging ===================
WANDB_PROJECT=Reasoning360-1.7B
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}-e6-s2-directcotstepjudge-frompretrain-0.2

# If RESUME_CKPT_DIR is not empty, resume from the checkpoint
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
fi


# =================== Ray start ===================
# ray stop at all nodes
#srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop
ray stop

sleep 10
# Remove existing Ray cluster
#srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
#srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
#    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
#    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --include-dashboard=True --temp-dir "$RAY_TMPDIR" --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    #echo "Starting WORKER $i at $node_i"
    #srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
    #    ${CONDA_BIN_PATH}ray start --address "$address_head" \
    #    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
    ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block & 
done
sleep 10

# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=16
train_prompt_mini_bsz=64  # model grad update batchsize

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Training config
# NOTE: sp_size and gen_tp are parallelism settings.
# sp_size: Sequence Parallelism size.
# gen_tp: Tensor Parallelism size for vLLM generation.
# For a 32B model on 8 GPUs, TP=2 is a reasonable starting point. Adjust if you have memory issues.
sp_size=1
gen_tp=2
gen_max_num_seqs=1024
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward, but note memory overflow
offload=True

# =================== Start RL training ===================
# Ensure your python environment (e.g., conda) is activated before running this script.
echo "Starting training..."
python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name="dapo_fsdp_config.yaml" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_seqs=${gen_max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    +actor_rollout_ref.model.torch_dtype=bfloat16 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_multi_process \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.log_val_generations=50 \
    trainer.resume_mode=auto