"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
Thanks to https://github.com/ganler/code-r1/blob/main/examples/data_preprocess/coder1.py
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule

import rich
import matplotlib.pyplot as plt
import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import (
    code_exec,
    remote_check_stdio,
    _ERROR_MSG_PREFIX,
)

from examples.data_preprocess.code.code_utils import *

WORKDING_DIR = os.path.join(os.environ.get("HOME"), "Reasoning360")


def kodcode():  # Thanks!!! to Zhangchen and Yueqin
    # library requirements?
    rich.print(Rule("Loading KodCode/KodCode-Light-RL-10K..."))
    dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    def make_map_fn(split):

        def process_fn(example, idx):
            reference_solution = example["solution"]
            test_code = "from solution import *\n" + example["test"].strip()
            # Filtering...
            # + block libs are used in reference solution and test code
            # + block usages are used in reference solution or test code
            # + filter out too long prompts
            # + filter out easy problems
            # + filter out failed unittests
            filter_info = {}
            for lib in BLOCK_LIBS:
                if (
                    f"import {lib}"
                    in reference_solution  # naive import detection; ast then detect would be better
                    or f"from {lib}" in reference_solution
                ):
                    print("===========Blocked lib in solution===========")
                    print(f"reference_solution:")
                    print(reference_solution)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for lib in BLOCK_LIBS:
                if f"import {lib}" in test_code or f"from {lib}" in test_code:
                    print("===========Blocked lib in test===========")
                    print(f"test_code:")
                    print(test_code)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in reference_solution:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in test_code:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            if (
                len(tokenizer.encode(example["question"])) > MAX_PROMPT_LENGTH - 200
            ):  # -200 for (approximately) the prompt template extra tokens
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "prompt_too_long", "detail": None},
                }
            if example["gpt_difficulty"] == "easy":
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "easy_problem", "detail": None},
                }
            succ, err = code_exec(code=reference_solution, pytest=test_code)
            if not succ:
                # The above code is using the `rich` library in Python to print a formatted message in the console.
                # The message is in red color and includes the value of `example['conversation_id']`.
                # rich.print(
                #     f"[bold red]Test code failed for {example['question_id']}"
                # )
                print("===========Unittest failed===========")
                print(f"reference_solution:")
                print(reference_solution)
                print(f"test_code:")
                print(test_code)
                print(f"err:")
                print(err)
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "failed_unittests", "detail": None},
                }

            prompt = f"Please solve the programming task below in Python. Code should wrapped in a markdown code block.\n\n{example['question'].strip()}"
            if example["test_info"]:
                prompt += f"\n\nNote that the output function should be {str(example['test_info']).strip()}."

            return {
                "data_source": "code-kodcode",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"pytest": test_code}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference_solution,
                    "prompt": prompt,
                    "dataset": "KodCode/KodCode-Light-RL-10K",
                    "question_subset": example["subset"],
                    "question_id": example["question_id"],
                    "gpt_difficulty": example["gpt_difficulty"],
                },
                "filter_info": None,
            }

        return process_fn

    dataset = dataset["train"].shuffle(seed=666)

    # Preprocess the dataset
    print("Executing tests to ensure correctness...")
    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        load_from_cache_file=False,
    )
    # Analyze the filter reasons
    filter_counts, filter_counts_fine_block_libs = {}, {}
    for entry in dataset:
        if entry["filter_info"] is None:
            continue
        filter_type = entry["filter_info"]["type"]
        if filter_type is not None:
            # Filter type distribution
            if filter_type not in filter_counts:
                filter_counts[filter_type] = 0
            filter_counts[filter_type] += 1
            # Filter detail distribution
            filter_detail = entry["filter_info"].get("detail", None)
            if filter_detail is not None:
                if filter_detail not in filter_counts_fine_block_libs:
                    filter_counts_fine_block_libs[filter_detail] = 0
                filter_counts_fine_block_libs[filter_detail] += 1
        # entry["filter_info"] = None

    print(f"Filtered samples from KodCode: {filter_counts}")

    plot_hist(
        filter_counts,
        file_path=os.path.join(WORKDING_DIR, "artifacts", "filter_counts.png"),
        title="Filter Sample Distribution from KodCode",
        xlabel="Filter Reason",
        ylabel="Count",
    )
    plot_hist(
        filter_counts_fine_block_libs,
        file_path=os.path.join(
            WORKDING_DIR, "artifacts", "filter_counts_fine_block_libs.png"
        ),
        title="Blocked Library Distribution from KodCode",
        xlabel="Blocked Library",
        ylabel="Count",
    )
    
    print(f"Before filtering, KodCode dataset size: {len(dataset)}")

    dataset = dataset.filter(lambda x: x["data_source"] is not None)
    print(f"Remaining samples from KodCode: {len(dataset)}")

    # pick random 50k examples for RL, otherwise it's too large
    # dataset = dataset.select(range(50000 + N_TESTSET_PER_DATASET))

    # Split into train and test
    # splits = dataset["train"].train_test_split(
    #     test_size=N_TESTSET_PER_DATASET, seed=666
    # )
    splits = dataset.train_test_split(test_size=N_TESTSET_PER_DATASET, seed=666)
    train_dataset = splits["train"].shuffle(seed=666)
    test_dataset = splits["test"]
    return train_dataset, test_dataset


# this dataset is super noisy and needs code execution to verify the tasks
def taco():
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified")["train"]
    

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return EMPTY_RETURN

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return EMPTY_RETURN

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return EMPTY_RETURN

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append(
                    "Also feel free to reuse/extend the following starter code:"
                )
                prompt_pieces.append(
                    f"```python\n{example['starter_code'].strip()}\n```"
                )

            ##
            ## Customization
            ##
            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]
                if source == "leetcode":
                    fn_name = "Solution()." + fn_name

                test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

                if source in ["leetcode", "hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = example["solutions"][-1] + "\n" + test_code
                if source in ["leetcode"]:
                    _check_test = PY_IMPORTS + _check_test

                succ, err = code_exec(_check_test)
                if not succ:
                    rich.print(f"[bold red]Test code failed for {source}")
                    print(_check_test)
                    print(err)
                    return EMPTY_RETURN
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(
                    oracle["inputs"], oracle["outputs"]
                )
                if len(stdin_list) == 0:
                    return EMPTY_RETURN

                with ThreadPoolExecutor(
                    max_workers=min(len(stdin_list), 8)
                ) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(
                            executor.submit(
                                remote_check_stdio,
                                example["solutions"][-1],
                                stdin,
                                stdout,
                            )
                        )
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and output.strip() == stdout.strip()
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for {source}")
                            print(example["solutions"][-1])
                            print(f"{exec_succ = }")
                            print(f"{stdin = }", f"{stdout = }")
                            if output.startswith(_ERROR_MSG_PREFIX):
                                print("output = \n", output)
                            else:
                                print(f"{output = }")
                            return EMPTY_RETURN

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (
                        example["solutions"][0] if example["solutions"] else ""
                    ),
                    "dataset": "likaixin/TACO-verified",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    ).filter(lambda x: x != EMPTY_RETURN)
    splits = dataset.train_test_split(
        test_size=max(1, min(N_TESTSET_PER_DATASET, len(dataset) * 0.1)), seed=666
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    for t in dataset:
        print(f"{t = }")
        t["extra_info"]["split"] = "test"

    print(f"Taco train set: {train_dataset}")
    print(f"Taco test set: {test_dataset}")

    return train_dataset, test_dataset
    train_dataset = dataset["train"]
    test_dataset = dataset["valid"][:N_TESTSET_PER_DATASET]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if "<image>" in example["description"]:
                print("Description includes image, skipping...")
                return EMPTY_RETURN

            stdin_list = (
                example["public_tests"]["input"]
                + example["private_tests"]["input"]
                + example["generated_tests"]["input"]
            )
            stdout_list = (
                example["public_tests"]["output"]
                + example["private_tests"]["output"]
                + example["generated_tests"]["output"]
            )

            stdin_list, stdout_list = minimize_stdio(
                stdin_list, stdout_list, max_n_tests
            )
            assert len(stdin_list) == len(stdout_list)
            if len(stdin_list) == 0:
                return EMPTY_RETURN

            prompt = (
                "Solve the programming task below in a Python markdown code block. "
                "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
                "produces outputs through STDOUT (like those in the 'Output' section)."
                f"\n\n{example['description'].strip()}"
            )
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {"inputs": stdin_list, "outputs": stdout_list}
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (
                        example["solutions"]["solution"][0]
                        if example["solutions"]["solution"]
                        else ""
                    ),
                    "dataset": "deepmind/code_contests",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    ).filter(lambda x: x != EMPTY_RETURN)
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), 
        with_indices=True,
        load_from_cache_file=False,
    )
    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    train_dataset = load_dataset("newfacade/LeetCodeDataset")["train"]
    test_dataset = load_dataset("newfacade/LeetCodeDataset")["test"]
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prefix = example["prompt"]
            
            prompt = example["query"]
            # remove the "### Answer: (use the provided format with backticks)" part
            prompt = prompt.replace("### Answer: (use the provided format with backticks)", "").strip()
            # adjust the "### Format: " part to be more readable
            prompt = prompt.replace("### Format: ", "### Format:\n")

            # Build test code (as before)
            test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
            # Extract the candidate solution (the original completion)
            solution = example["completion"]
        
            # Combine all code pieces into a single file to execute.
            full_code = f"{prefix}\n{solution}\n{test_code}"
            
            # Validate that the candidate solution passes the tests
            # 20s timeout as some leetcode tests are slow
            succ, err = code_exec(full_code, timeout=20)
            
            if not succ:
                print("===========Test code failed for LeetCodeDataset===========")
                print("Question:", example["meta"]["question_title"])
                print("Error:", err)
                # Skip the example if the test code fails
                return EMPTY_RETURN

            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({
                        "functional": test_code
                    }),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": solution,
                    "prompt": prompt,
                    "prefix": prefix,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    # filter out empty examples ("reward_model" is None)
    train_dataset = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    print(f"Leetcode2k train set: {train_dataset}")
    print(f"Leetcode2k test set: {test_dataset}")
    return train_dataset, test_dataset


def humaneval():
    rich.print(Rule("Loading OpenAI HumanEval..."))
    dataset = load_dataset("openai_humaneval")["test"]
    print("HumanEval dataset:", dataset)
    
    def process_fn(example, idx):
        # HumanEval's prompt already contains the function signature and docstring
        prompt = (
            "Write a complete, self-contained Python solution to the following problem. "
            "Your solution must include all necessary imports and the full function definition including "
            "the signature exactly as specified. Do not modify the function signature or docstring.\n\n"
            f"```python\n{example['prompt'].strip()}\n```"
        )
        
        # Extract test code
        test_code = example['test']
        entry_point = example['entry_point']
        
        # Validate that the canonical solution passes the tests
        solution = example['canonical_solution']
        
        # Combine the prompt code + solution + test code to verify it works
        full_code = f"{example['prompt']}\n{solution}\n{test_code}\n\ncheck({entry_point})"
        
        succ, err = code_exec(full_code)
        if not succ:
            print(f"Error in canonical solution for task {example['task_id']}: {err}")
            return EMPTY_RETURN
        
        return {
            "data_source": "code",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    {"functional": f"{test_code}\n\ncheck({entry_point})"}
                ),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "reference": solution,
                "prompt": prompt,
                "dataset": "openai_humaneval",
                "task_id": str(example["task_id"]),
            },
        }
    
    test_dataset = dataset.map(
        function=process_fn, 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    
    # Return empty train dataset and test dataset
    empty_train = datasets.Dataset.from_dict({
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": []
    }) if len(test_dataset) > 0 else datasets.Dataset.from_dict({})
    
    print(f"HumanEval test set: {test_dataset}")
    return empty_train, test_dataset

def mbpp():
    rich.print(Rule("Loading MBPP dataset..."))
    dataset = load_dataset("google-research-datasets/mbpp")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # rewrite the task_id as it is int
            example["task_id"] = "MBPP/" + str(example["task_id"])
            
            # Create prompt
            prompt = (
                f"{example['text']}\n\n"
                f"Your solution should be a complete, self-contained function in a markdown code block. "
                f"Make sure your solution passes the following test cases:\n"
            )
            
            # Construct test code
            test_code = ""
            if example.get('test_setup_code'):
                test_code += example['test_setup_code'] + "\n\n"
            
            # Add all test assertions
            for assertion in example['test_list'] + example.get('challenge_test_list', []):
                test_code += assertion + "\n"
            
            # Add test cases to prompt
            prompt += f"```python\n{test_code}```"
            prompt += "\n\nPlease do not include the test cases in your solution."
            
            # Validate that the canonical solution passes the tests
            solution = example['code']
            full_code = f"{solution}\n\n{test_code}"
            
            succ, err = code_exec(full_code)
            if not succ:
                print(f"Error in canonical solution for task {example['task_id']}: {err}")
                return EMPTY_RETURN
            
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {"functional": test_code}
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": solution,
                    "prompt": prompt,
                    "dataset": "mbpp",
                    "task_id": str(example["task_id"]),
                },
            }
        
        return process_fn
    
    # Process train and test splits
    train_dataset = dataset["train"].map(
        function=make_map_fn("train"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    test_dataset = dataset["test"].map(
        function=make_map_fn("test"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    print(f"MBPP train set: {train_dataset}")
    print(f"MBPP test set: {test_dataset}")
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=os.path.join(WORKDING_DIR, "data"))
    parser.add_argument(
        "--dataset_names", default="kodcode", help="comma separated dataset names"
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_names = args.dataset_names.split(",")

    train_datasets = []
    test_datasets = []

    dataset_map = {
        "kodcode": kodcode,
        "taco": taco,
        "leetcode2k": leetcode2k,
        "humaneval": humaneval,
        "mbpp": mbpp,
    }
    dataset_makes = [dataset_map[name] for name in dataset_names]
    names = "-".join([make.__name__ for make in dataset_makes])

    for make in dataset_makes:
        train, test = make()
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(
        root_dir, f"codegen-{round(len(train_dataset) / 1000)}k-{names}"
    )
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))