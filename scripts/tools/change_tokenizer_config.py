import json
import os
import shutil
import argparse
from transformers import AutoTokenizer

def modify_tokenizer_config(input_path, output_path):
    """Create a modified copy of the tokenizer config in a new directory"""
    # Copy the entire model directory to the new location
    print(f"Copying {input_path} to {output_path}")
    shutil.copytree(input_path, output_path, dirs_exist_ok=True)
    print(f"Copied {input_path} to {output_path}")
    # Work with the config in the new directory
    config_path = os.path.join(output_path, "tokenizer_config.json")
    
    # Read the original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Change the EOS token
    config['eos_token'] = '<|im_end|>'
    
    # 2. Modify the chat template to change default system prompt
    chat_template = config['chat_template']
    
    # Replace the default system prompt
    new_system_prompt = "You are a helpful assistant. To answer a query from the user, please first thinks through the question step-by-step inside <think>...</think>, then provides the final response to user."
    
    # Update the system prompt in both conditions (with and without tools)
    chat_template = chat_template.replace(
        "You are a helpful assistant.",
        new_system_prompt
    )
    
    # 3. Add <think> tag after assistant generation prompt
    chat_template = chat_template.replace(
        "{{- '<|im_start|>assistant\\n' }}",
        "{{- '<|im_start|>assistant\\n<think>' }}"
    )
    
    config['chat_template'] = chat_template
    
    # Save the modified config in the new directory
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_path

def test_chat_template(model_path):
    """Test the modified chat template"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Example conversation
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nGenerated prompt:")
    print(prompt)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Modify tokenizer config with think tags and copy to new directory'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input model directory path'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory path (default: input_path-think)',
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    # If output path not specified, create default
    if args.output is None:
        args.output = f"{args.input}-think"
    
    print(f"Modifying tokenizer config for {args.input} and saving to {args.output}")
    new_path = modify_tokenizer_config(args.input, args.output)
    print(f"Modified model saved to: {new_path}")
    
    # Test the modified chat template
    test_chat_template(new_path)