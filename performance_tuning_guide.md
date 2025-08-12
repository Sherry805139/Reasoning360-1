Optimizing training speed for large models like the 70B long-context model is a battle against two constraints: GPU memory and computational throughput. This guide explains the key parameters in veRL that allow you to manage these constraints effectively.

## 1. Preliminaries

Before diving into individual parameters, there is a crucial formula of global (or mini-)batch size to keep in mind.

$$
\text{mini\_batch\_size} = \text{micro\_batch\_size} \times \text{grad\_accum\_steps} \times \text{data\_parallel\_size}
$$

- **Mini-batch size**: The true batch size that the optimizer sees. Affects ML stability and convergence.
- **Micro-batch size**: What a single GPU processes in one forward/backward pass. Affects GPU memory usage and utilization.
- **Gradient accumulation**: How many micro-batch gradients to sum up before updating the model, usually to achieve a larger batch size.
- **Data parallel size (DP)**: The number of parallel groups working on different data replicas.


## 2. Training

- **Mini-batch size(`actor.ppo_mini_batch_size`)**[tag: ML, system]: The mini (or global) batch size to update the model's gradients. This is the conventional mini (or global) batch size in classical ML and large-scale LLM (pre-)training.
    - From an ML perspective, a larger batch size leads to more stable gradient estimates (lower variance), which is especially beneficial for LLM training. However, too large a mini-batch size can hurt model generalization by converging to sharp minima, slow down convergence in terms of number of parameter updates (fewer updates per epoch). When using online RL algorithms such as PPO, setting `actor.ppo_mini_batch_size` close to or even equal to `data.train_prompt_bsz` can improve “on-policyness,” thereby reducing policy lag and potentially leading to more stable and consistent updates.
    - From a system perspective, we need to consider GPU memory limits and training speed. A larger mini-batch size, when achievable, generally leads to faster training by better utilizing the GPU's computational resources. This speedup is most obvious when no gradient accumulation is needed. Even with gradient accumulation, a larger mini-batch size can be faster because it amortizes the cost of the optimizer step over more data <toverify>.
- **Micro-batch size(`actor.ppo_micro_batch_size_per_gpu`**and deprecated **`actor.ppo_micro_batch_size`)**[tag: system]: The actual batch size processed per GPU (or DP) in a single forward/backward pass. Unlike Mini-batch size, Micro-batch size is purely a system optimization parameter and does not directly affect ML performance. Its primary role is to manage memory usage while maximizing GPU utilization. Finding the optimal micro-batch size is tricky: it depends on the training sequence length, model sharding strategy (e.g., FSDP), and model parallelization scheme (e.g., Megatron). Also note, the `per_gpu` naming is a bit imprecise, it actually means `per_data_parallel` . E.g., if you enable sequence parallel, multiple GPUs in one DP handle one sample.
    
    *Best practice:* Start with a micro-batch size of 1. If there are no out-of-memory (OOM) errors, incrementally increase it to find the sweet spot that maximizes throughput. If you encounter OOM, consider more aggressive sharding or parallelism strategies like sequence parallelism before reducing the micro-batch size. Another option is to use `use_dynamic_bsz`.
    
- **Dynamic batch size(`actor.use_dynamic_bsz`)**[tag: system]: If true, this setting packs multiple sequences into a single, long sequence up to a specified token limit and **ignores the specified micro-batch size**. Instead of tuning the micro-batch size, you tune `actor.ppo_max_token_len_per_gpu`. Under the hood, it uses sequence packing to arrange sequences into a single batch, minimizing padding tokens. This technique saves redundant GPU compute that would otherwise be wasted on padding tokens.
    
    *Best practice*: Enable it for workloads with variable sequence lengths.
    
- **Sequence parallel(`actor.ulysses_sequence_parallel_size`** in FSDP or **`actor.megatron.context_parallel_size`** in Megatron**)**: The Transformer architecture's self-attention mechanism leads to activation memory scaling (near) quadratically with context length ($O(N^2)$). In long-context regimes (e.g., 32k sequence length), this activation memory becomes a primary cause of OOM errors. Sequence Parallelism (SP) addresses this by sharding the input sequence across GPUs within a data-parallel group. For example, setting `SP=4` on a 32k sequence would result in each of the 4 GPUs processing an 8k sub-sequence. The trade-off is that your effective data-parallel size is reduced by a factor of SP, which can slow down training due to increased communication for the attention mechanism's all-gather operations.
    
    *Best practice*: Set `SP > 1` when OOM errors are primarily caused by long context lengths.
    
- **Gradient checkpointing(`enable_gradient_checkpointing`)**[tag: system]: During the backward pass, gradients are computed using activations from the forward pass, which are typically stored in GPU memory. Instead of storing all activations, gradient checkpointing saves only a subset and recomputes the others on-the-fly during the backward pass. This can drastically reduce memory consumption at the cost of computation time for the backward pass.

- **Parameter offload(`param_offload`, `grad_offload`, `optimizer_offload`)**[tag: system]: This technique frees up GPU memory by moving model parameters or optimizer states to CPU. During computation, the necessary data is transferred back to the GPU. This can save a significant amount of GPU memory but introduces a substantial communication bottleneck between the CPU and GPU.

**FSDP-specific**

- **FSDP size(`fsdp_size=-1`)**[tag: system]: The `fsdp_size` controls the number of GPUs in each FSDP group. `fsdp_size=-1` (default): FSDP will shard the model parameters, gradients, and optimizer states **across all available training GPUs**. `fsdp_size > 1`: Specifies a custom number of GPUs per FSDP group.

**Megatron-specific**

- **Tensor parallel(`megatron.tensor_model_parallel_size`)**[tag: system]: Tensor Parallelism (TP) is an *intra-layer* parallelism technique that shards individual weight matrices within Transformer layers (e.g., in MLP and self-attention blocks) across multiple GPUs. This significantly reduces the memory footprint of the model parameters, grad optimizer states, and activation on each GPU. However, it requires high-bandwidth communication (all-reduce operations) within the TP group after each parallel computation.
- **Pipeline parallel(`megatron.pipeline_model_parallel_size`)**[tag: system]: Pipeline Parallelism (PP) is an *inter-layer* parallelism technique that partitions the model's layers into sequential stages, with each stage assigned to a different GPU. This reduces memory by requiring each GPU to store only a fraction of the model's layers and their activations.

## 3. Rollout (vLLM)

- **Generation batch size(`data.train_prompt_bsz`)**: This is the number of unique prompts sent to the vLLM engine for generation in a single batch. This is distinct from the training mini-batch size. A larger generation batch size can increase throughput by better utilizing the GPU and reducing the times of switch of rollout and train phases, but introduces the risk of “off-policy” if your training doesn’t support large enough mini-batch size (`actor.ppo_mini_batch_size`).
- **Tensor parallel(`rollout.tensor_model_parallel_size`)**: Similar to training, this shards the model's weights across multiple GPUs to serve models that are too large for a single GPU. For inference, TP is crucial for both fitting the model in memory and for increasing throughput.
- **KV cache(`rollout.gpu_memory_utilization`)**: The Key-Value (KV) cache is crucial for faster inference and the primary memory consumer during LLM inference. It stores the key and value states for all previously generated tokens in a sequence to prevent costly recomputation. `gpu_memory_utilization` in vLLM controls the fraction of GPU memory pre-allocated for the KV cache. A higher value (e.g., 0.90) allows vLLM to handle more concurrent requests and/or longer sequences efficiently, but leaves less memory for other processes.
- **Max token length per GPU(`infer_ppo_max_token_len`)**: Max tokens to be processed in the forward computation. Similar to `actor.ppo_max_token_len_per_gpu` in training.

## 4. Case Studies

An example 70B long-cot training, using distilled-r1-Llama3.1-70B as example, is run `sbatch scripts/example_multinode_rl_llama3.1_70b_distill_megatron.sh` using 32 nodes with training TP=8, PP=2, SP=8, rollout TP=4. The parameters are tuned (e.g., we find PP doesn't need to be high, 2 is enough), but not optmized.