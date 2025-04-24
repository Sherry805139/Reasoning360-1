# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score.dsp1.prover.lean.verifier import Lean4ServerScheduler  # added by reasoning360
import torch
from collections import defaultdict


class ProverLean4RewardManager:
    """Asynchronous reward manager using Lean4ServerScheduler."""

    def __init__(self, tokenizer, num_cpus=64, num_examine: int = 1, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # initialize Lean4 scheduler
        self.lean4_scheduler = Lean4ServerScheduler(
            max_concurrent_requests=num_cpus,
            timeout=300,
            memory_limit=200,
            name="lean4_hf_verifier_stability"
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Batch verification: submit all snippets at once, then fetch outputs.
        Returns reward_tensor and optional reward_extra_info.
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # Step 1: prepare all full_snippets
        full_snippets = []
        meta = []  # store (data_source, ground_truth, prompt_str, resp_len)
        for i in range(batch_size):
            item = data[i]
            prompts_id = item.batch["prompts"]
            attn = item.batch["attention_mask"]
            prompt_len = attn.sum(dim=-1).item()
            prompt_str = self.tokenizer.decode(
                prompts_id[0, :prompt_len], skip_special_tokens=True)

            responses_id = item.batch["responses"]
            resp_len = attn[0, prompt_len:].sum().item()
            response_str = self.tokenizer.decode(
                responses_id[0, :resp_len], skip_special_tokens=True)
            
            statement = item.non_tensor_batch["statement"]
            ground_truth = statement + "\n" + item.non_tensor_batch["response"]

            snippet = statement + "\n" + response_str
            full_snippets.append(snippet.replace("\\n", "\n"))
            
            meta.append((ground_truth, prompt_str, response_str, resp_len))

        # Step 2: submit all requests asynchronously
        request_ids = self.lean4_scheduler.submit_all_request(full_snippets)
        # Step 3: get all outputs
        outputs = self.lean4_scheduler.get_all_request_outputs(request_ids)

        # Step 4: process outputs
        for i, out in enumerate(outputs):
            data_source, ground_truth, prompt_str, response_str, resp_len = meta[i]
            passed = bool(out.get("pass", False))
            score = 1.0 if passed else 0.0
            reward_tensor[i, resp_len - 1] = score
            reward_extra_info["score"].append(score)
            reward_extra_info["lean_out"].append(out)

            # debugging print
            if reward_extra_info.get(data_source, []).count(score) < self.num_examine:
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print(f"[score] {score}")
                print("[lean_out]", out)

        if return_dict:
            return {"reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info}
        return reward_tensor

    def close(self):
        """Close the scheduler to free resources."""
        self.lean4_scheduler.close()



if __name__ == "__main__":
    from datasets import load_dataset
    # Simple test for ProverLean4RewardManager
    from transformers import AutoTokenizer
    from verl import DataProto
    import torch
    from tensordict import TensorDict
    dataset = load_dataset("deepseek-ai/DeepSeek-Prover-V1", split="train")

    # Initialize tokenizer and manager
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    manager = ProverLean4RewardManager(tokenizer, num_cpus=2, num_examine=1)

    # Prepare a dummy proof example
    for i, example in enumerate(dataset):
        snippet = example["header"] + example["formal_statement"] + example["formal_proof"]
        snippet = snippet.replace("\\n", "\n")

    question = '''
    import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat

    theorem thm_0 : 
    let h := (3 : ℝ) / 2;
    let n := 5;
    h^n ≤ 0.5 → false := by
    '''
    proof = '''
  intro h n
  norm_num [h, n]
'''

    # Tokenize prompt and response
    prompt_ids = tokenizer.encode(question, return_tensors="pt")
    response_ids = tokenizer.encode(proof, return_tensors="pt")

    # Build a simple attention mask
    combined = torch.cat([prompt_ids, response_ids], dim=-1)
    attention_mask = torch.ones_like(combined)

    # Create batch dicts
    batch_td = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": response_ids,
            "attention_mask": attention_mask,
        },
        batch_size=prompt_ids.shape
    )

    # Non-tensor metadata
    non_tensor = [
        {
            "reward_model": {"ground_truth": proof},
            "data_source": "test_source"
        }
    ]

    # Construct DataProto and call manager
    data_proto = DataProto(batch_td, non_tensor)
    result = manager(data_proto, return_dict=True)

    print("Reward Tensor:", result["reward_tensor"])
    print("Reward Extra Info:", result["reward_extra_info"])

    # Clean up
    manager.close()
