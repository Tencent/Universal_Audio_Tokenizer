import logging
import random
from datasets import load_dataset, concatenate_datasets


def prepare_custom_dataset_v1(data_file, rank=None, world_size=None, cache_dir=None,
                                  debug_mode=False, is_test_data=False):
    """
    Read file with parquet paths, merge all parquet data into one dataset
    Use memory mapping to read parquet files to reduce memory usage

    Args:
        data_file: File path containing parquet file paths
        rank: Current process rank (starts from 0)
        world_size: Total number of processes
    Returns:
        DatasetDict: Dataset object containing training and validation sets
    """

    # Read file path list
    with open(data_file, 'r') as f:
        parquet_files = [line.strip() for line in f.readlines()]
    random.seed(42)
    random.shuffle(parquet_files)

    # If distributed training, partition file list according to rank
    if is_test_data:
        parquet_files = parquet_files
    else:
        # Original file allocation logic
        files_per_process = len(parquet_files) // world_size
        start_idx = rank * files_per_process
        end_idx = start_idx + files_per_process if rank < world_size - 1 else len(parquet_files)
        parquet_files = parquet_files[start_idx:end_idx]
        if rank == 0:
            logging.info(f"Process {rank}: processing {len(parquet_files)} files")

    # Read all parquet files and merge
    datasets = []
    for idx, parquet_path in enumerate(parquet_files):
        if rank == 0:
            logging.info(f"Process {rank} | {idx}/{len(parquet_files)}: Loading parquet file: {parquet_path}")
        dataset = load_dataset('parquet', data_files=parquet_path, split='train', cache_dir=cache_dir)
        if debug_mode:
            dataset = dataset.select(range(2))
        datasets.append(dataset)

    # Concatenate all datasets
    assert len(datasets) > 0, "No datasets to concatenate"
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset


def prepare_custom_dataset(data_file, rank=None, world_size=None, cache_dir=None, debug_mode=False, is_test_data=False):
    """
    Read file with parquet paths, merge all parquet data into one dataset
    Use memory mapping to read parquet files to reduce memory usage

    Args:
        data_file: File path containing parquet file paths
        rank: Current process rank (starts from 0)
        world_size: Total number of processes
    Returns:
        DatasetDict: Dataset object containing training and validation sets
    """
    with open(data_file, 'r') as f:
        all_parquet_files = [line.strip() for line in f.readlines()]

    if not all_parquet_files:
        logging.warning("No parquet files found in the data_file.")
        # Return empty dataset or raise error as needed
        return None     # Or Dataset.from_dict({})

    random.seed(42)
    random.shuffle(all_parquet_files)

    # File list assigned to current rank
    files_for_this_rank = []
    if is_test_data:
        files_for_this_rank = all_parquet_files
    else:
        total_files = len(all_parquet_files)
        files_per_process = total_files // world_size
        remainder = total_files % world_size

        start_idx = rank * files_per_process + min(rank, remainder)
        end_idx = start_idx + files_per_process + (1 if rank < remainder else 0)

        files_for_this_rank = all_parquet_files[start_idx:end_idx]

    if not files_for_this_rank:
        logging.info(f"Process {rank} (world_size {world_size}): No files assigned to this rank. Returning None.")
        return None     # Or Dataset.from_dict({})

    if rank == 0 or rank is None:   # Print info in main process or non-distributed
        logging.info(f"Process {rank} (world_size {world_size}): Assigned {len(files_for_this_rank)} files.")
        if files_for_this_rank:
            logging.info(f"Process {rank} | First file: {files_for_this_rank[0]}, "
                         f"Last file: {files_for_this_rank[-1]}")

    num_loader_proc = 32

    logging.info(f"Process {rank}: Loading {len(files_for_this_rank)} parquet files "
                 f"using num_proc={num_loader_proc}...")

    try:
        combined_dataset_for_rank = load_dataset(
            'parquet',
            data_files={'train': files_for_this_rank},
            split='train',
            cache_dir=cache_dir,
            num_proc=num_loader_proc,
            # keep_in_memory=False, # For large files, usually keep False
        )
    except Exception as e:
        logging.error(f"Process {rank}: Error loading dataset with files: {files_for_this_rank[:3]}... Error: {e}")
        return None

    if rank == 0 or rank is None:
        logging.info(f"Process {rank}: Successfully loaded its assigned files "
                     f"into a dataset with {len(combined_dataset_for_rank)} examples.")

    return combined_dataset_for_rank
