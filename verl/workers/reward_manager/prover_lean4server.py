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

import re, os, datetime, torch
from collections import defaultdict
from difflib import SequenceMatcher
from verl import DataProto
from verl.utils.reward_score.dsp1.prover.lean.verifier import Lean4ServerScheduler

# ---------- 工具函数 ----------
_theorem_name_re = re.compile(r"\btheorem\s+(\w+)", re.IGNORECASE)
_ws_re           = re.compile(r"\s+")

def _normalize(t: str) -> str:
    t = re.sub(r"--.*", "", t)                # 行注释
    t = re.sub(r"/-.*?-/", "", t, flags=re.S) # 块注释
    return _ws_re.sub("", t)

def statement_solved(statement: str, proof: str) -> bool:
    m = _theorem_name_re.search(statement)
    if m and not re.search(rf"\btheorem\s+{re.escape(m.group(1))}\b", proof):
        return False
    s_norm, p_norm = _normalize(statement), _normalize(proof)
    if s_norm and s_norm in p_norm:
        return True
    return SequenceMatcher(None, s_norm, p_norm).quick_ratio() > 0.9

def extract_lean4proof(text: str) -> str:
    blocks = re.findall(r"```lean\s*\n([\s\S]*?)```", text, re.I)
    return blocks[-1].strip("\n") if blocks else ""

# ---------- 主类 ----------
class ProverLean4RewardManager:
    def __init__(self, tokenizer, num_cpus=32, num_examine=1,
                 save_lean_dir="./lean4_snippets", **_):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_re = re.compile(r"```lean[\s\S]*?```", re.I)
        self.lean4_scheduler = Lean4ServerScheduler(
            max_concurrent_requests=num_cpus, timeout=300, memory_limit=200,
            name="lean4_hf_verifier_stability")
        self.save_lean_dir = save_lean_dir
        os.makedirs(save_lean_dir, exist_ok=True)

    # --------- 主调用 ---------
    def __call__(self, data: DataProto, return_dict=False):
        batch = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        extra = defaultdict(list)

        full_snips, meta = [], []

        # Step-1 解析 response
        for i in range(batch):
            item = data[i]
            attn      = item.batch["attention_mask"]
            promptlen = int(attn.sum())
            resplen   = int(attn[promptlen:].sum())
            if resplen == 0:
                extra["format_score"].append(0.0)
                extra["statement_hit"].append(0.0)
                continue

            resp_ids = item.batch["responses"]
            resp_str = self.tokenizer.decode(resp_ids[:resplen], skip_special_tokens=True)

            fmt_ok = bool(self.format_re.search(resp_str))
            extra["format_score"].append(float(fmt_ok))
            # 只有 resplen>0 时才安全写
            reward_tensor[i, resplen-1] = float(fmt_ok)

            proof_code = extract_lean4proof(resp_str)
            stmt       = item.non_tensor_batch["statement"]
            hit_stmt   = statement_solved(stmt, proof_code)
            extra["statement_hit"].append(float(hit_stmt))

            if not (proof_code and hit_stmt):
                continue

            full_snips.append(proof_code)
            meta.append((i, stmt, proof_code, resplen))

            if i < self.num_examine:
                print("=== prompt ===\n", stmt)
                print("=== response ===\n", resp_str.splitlines()[:40])

        # Step-2 Lean4 校验
        outs = self.lean4_scheduler.get_all_request_outputs(
            self.lean4_scheduler.submit_all_request(full_snips))

        # Step-3 通过则加分 + 保存
        for j, out in enumerate(outs):
            i, stmt, proof_code, resplen = meta[j]
            passed = bool(out.get("pass")) and bool(out.get("complete"))
            if passed:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                with open(os.path.join(self.save_lean_dir, f"{ts}_idx{i}.lean"),
                          "w", encoding="utf-8") as f:
                    f.write(stmt + "\n" + proof_code)
                reward_tensor[i, resplen-1] += 1.0

        # 结果打包
        if return_dict:
            total   = reward_tensor.sum(dim=1).tolist()
            fmt     = extra["format_score"]
            extra["correctness_score"] = [t - f for t, f in zip(total, fmt)]
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor

    def close(self):
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
