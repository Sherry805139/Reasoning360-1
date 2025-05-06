import datasets
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Callable, Any, Dict

# Global variables for multiprocessing
IS_EQUAL_FUNC = None

def default_is_equal(text1: str, text2: str) -> bool:
    """Default equality check function."""
    return text1 in text2 or text2 in text1

def is_unique_item(item: Tuple[int, str], indexed_items: List[Tuple[int, str]]) -> bool:
    """Check if an item is unique compared to previous items."""
    idx, content = item
    for prev_idx, prev_content in indexed_items:
        if prev_idx >= idx:
            break
        if IS_EQUAL_FUNC(content, prev_content):
            return False
    return True

def process_item(item: Tuple[int, str], 
                all_indexed_items: List[Tuple[int, str]]) -> bool:
    """Process single item."""
    curr_idx, _ = item
    previous_items = [
        (i, content) for i, content in all_indexed_items 
        if i < curr_idx
    ]
    return is_unique_item(item, previous_items)

def deduplicate_datasets(
    dataset_configs: List[dict],
    content_extractors: Dict[str, Callable[[Any], str]],
    is_equal_func: Callable[[str, str], bool] = default_is_equal,
    num_processes: int = mp.cpu_count() - 2
) -> List[datasets.Dataset]:
    """
    Deduplicate multiple datasets globally.
    
    Args:
        dataset_configs: List of dicts with keys:
            - 'name': dataset name
            - 'split': dataset split
        content_extractors: Dict mapping dataset names to their content extractors
        is_equal_func: Function to determine if two items are equal
        num_processes: Number of processes for parallel processing
    """
    # Set global variables for multiprocessing
    global IS_EQUAL_FUNC
    IS_EQUAL_FUNC = is_equal_func

    # Load all datasets and extract contents
    print("Loading datasets and extracting contents...")
    loaded_datasets = []
    indexed_contents = []
    global_idx = 0
    
    for config in dataset_configs:
        dataset = datasets.load_dataset(config['name'])[config['split']]
        extractor = content_extractors[config['name']]
        
        # Extract contents for this dataset
        contents = [extractor(item) for item in tqdm(dataset, desc=f"Extracting from {config['name']}")]
        
        # Store dataset and its contents
        loaded_datasets.append((config['name'], dataset))
        indexed_contents.extend([
            (global_idx + i, content) for i, content in enumerate(contents)
        ])
        global_idx += len(dataset)

    # Process in parallel
    print("Processing items...")
    with mp.Pool(num_processes) as pool:
        process_func = partial(process_item, all_indexed_items=indexed_contents)
        is_unique = list(tqdm(
            pool.imap(process_func, indexed_contents, chunksize=100),
            total=len(indexed_contents),
            desc="Deduplicating"
        ))

    # Add dedup column to each dataset
    print("Adding dedup column to datasets...")
    deduplicated_datasets = []
    start_idx = 0
    for dataset_name, dataset in loaded_datasets:
        end_idx = start_idx + len(dataset)
        dedup_mask = is_unique[start_idx:end_idx]
        deduplicated_datasets.append(dataset.add_column("is_unique", dedup_mask))
        start_idx = end_idx

    return deduplicated_datasets

if __name__ == "__main__":
    # Example usage
    configs = [
        {'name': 'SDSB/deduplicated_dapo_dataset', 'split': 'train'},
        {'name': 'SDSB/or1_dataset', 'split': 'train'}
    ]
    
    # Define dataset-specific content extractors for deduplication
    content_extractors = {
        'SDSB/deduplicated_dapo_dataset': lambda x: (
            x['original_question'].lower()
        ),
        'SDSB/or1_dataset': lambda x: (
            x['original_question'].lower()
        )
    }
    
    # Define the is_equal_func as a lambda function
    is_equal_func=lambda x, y: x in y or y in x
    
    
    # Example with global deduplication
    deduplicated_datasets = deduplicate_datasets(
        dataset_configs=configs,
        content_extractors=content_extractors,
        is_equal_func=is_equal_func
    )

    # save the new dataset here
    for dataset, config in zip(deduplicated_datasets, configs):
        # print stats
        print(f"Dataset {config['name']} has {len(dataset)} items")
        print(f"Dataset {config['name']} has {len(dataset.filter(lambda x: x['is_unique']))} unique items")
        dataset.save_to_disk(f"{config['name'].split('/')[-1]}_globally_deduplicated")