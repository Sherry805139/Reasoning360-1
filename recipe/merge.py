import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load both models
model_a = AutoModelForCausalLM.from_pretrained("/mnt/sharefs/users/chengqian.gao/checkpoints/PALU/756464-math-1.5b-dalu-th8-alpha5-16k-20epoch/global_step_440/actor")
model_b = AutoModelForCausalLM.from_pretrained("/mnt/sharefs/users/chengqian.gao/checkpoints/PALU/756464-math-1.5b-dalu-th8-alpha5-16k-20epoch/global_step_420/actor")

tokenizer = AutoTokenizer.from_pretrained("/mnt/sharefs/users/chengqian.gao/checkpoints/PALU/756464-math-1.5b-dalu-th8-alpha5-16k-20epoch/global_step_440")
# Average the weights
for name, param in model_a.named_parameters():
    param.data = 0.5 * param.data + 0.5 * model_b.state_dict()[name]

# Save the merged model
model_a.save_pretrained("/mnt/sharefs/users/chengqian.gao/checkpoints/PALU/756464-math-1.5b-dalu-th8-alpha5-16k-20epoch/global_step_420", torch_dtype=torch.bfloat16)
tokenizer.save_pretrained('/mnt/sharefs/users/chengqian.gao/checkpoints/PALU/756464-math-1.5b-dalu-th8-alpha5-16k-20epoch/global_step_420')