# !/bin/bash

set -x

WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data


# need to process the data and specify the train reward metric
# e.g.,
# python bigmath_preview_filtered_mar21.py --train-reward-metric math_llm_judge --suffix _llm_judge

train_files="['data/train/math__bigmath_preview_filtered_mar21_llm_judge_157.3k.parquet']"
test_files="['data/test/math__math_500.parquet']"

# first download the model from huggingface
# huggingface-cli download Qwen/Qwen2.5-3B --local-dir ~/hf_models/Qwen2.5-3B

# serve vllm and update the ip address
# vllm serve Qwen/Qwen2.5-32B-Instruct --host 0.0.0.0 -tp 8 --port 8000

# change the ip below to the node of LLM judge
export MATH_LLM_JUDGE_URL=http://176.56.200.81:8000/v1/chat/completions

BASE_MODEL=${HOME}/hf_models/Qwen2.5-3B

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=${BASE_MODEL##*/}-Math-LLM-Judge

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=32 \
    reward_model.reward_manager=prime \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${BASE_MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.adv_estimator=grpo \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.val_generations_to_log_to_wandb=50 \
    trainer.total_epochs=15 $@
