SHARED_DATA_PATH=/mnt/sharefs/users/zhuojun.cheng
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/guru_data/train/guru92k_release_0603
TEST_DATA_DIR=${SHARED_DATA_PATH}/guru_data/test/online
export RAY_DEDUP_LOGS=0

export CUDA_DIR=/mnt/sharefs/software/DeepEP/cuda-12-4
export CUDNN_DIR=/mnt/weka/home/varad.pimpalkhute/cudnn-linux-x86_64-9.11.0.98_cuda12-archive

export NVTE_CUDA_INCLUDE_DIR=$CUDA_DIR/include

export CUDNN_LIBRARY=$CUDNN_DIR/lib/libcudnn.so
export CUDNN_INCLUDE_DIR=$CUDNN_DIR/include

export LD_LIBRARY_PATH=$CUDA_DIR/lib:$CUDNN_DIR/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_DIR/bin:$PATH
export CMAKE_PREFIX_PATH=$CUDA_DIR:$CUDNN_DIR:$CMAKE_PREFIX_PATH

python -m async_rl.main \
    --config-path "./config" \
    --config-name "train_config.yaml" \
    data.train_files=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet \
    data.val_files=${TEST_DATA_DIR}/math__math_500.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct