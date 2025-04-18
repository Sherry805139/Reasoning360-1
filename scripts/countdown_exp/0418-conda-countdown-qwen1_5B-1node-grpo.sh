#!/bin/bash
#SBATCH --job-name=rl_countdown
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/countdown-grpo-%j.out
#SBATCH --error=slurm/countdown-grpo-%j.err
#SBATCH --exclusive
#SBATCH --time=24:00:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node=${nodes[0]}

echo "${nodes[@]}"
for host in ${nodes[@]}; do
    echo "Checking node: $host"
    srun --nodes=1 --ntasks=1 --nodelist=$host \
         ~/Reasoning360/scripts/check_gpu.sh

    if [ $? -ne 0 ]; then
        echo "ERROR: Found GPU usage by other users on $host. Exiting."
        exit 1
    fi
done

echo "=== No leftover GPU usage found on all allocated nodes. ==="
echo "Proceeding with the main job..."

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# Experiment config
WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data
countdown_train_path=${DATA_DIR}/countdown/train.parquet
countdown_test_path=${DATA_DIR}/countdown/test.parquet

train_files="['$countdown_train_path']"
test_files="['$countdown_test_path']"
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# BASE_MODEL=Qwen/Qwen2.5-3B
BASE_MODEL=Qwen/Qwen2.5-1.5B

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 20))
enable_overlong_buffer=False
overlong_buffer_len=0
overlong_penalty_factor=0.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=64
train_prompt_mini_bsz=32

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


WANDB_PROJECT=TinyZero
WANDB_EXPERIMENT_NAME=taylor-countdown-grpo-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export worker_num=$SLURM_NNODES
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0
export HYDRA_FULL_ERROR=1

"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=2048 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    acror_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console', 'wandb'] \
    +trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.val_generations_to_log_to_wandb=10 \
    trainer.resume_mode=disable