import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from src.train.training_args import TrainingArguments


def create_train_dataloader(dataset, args: TrainingArguments,
                            data_collator, base_seed=42, rank=0) -> DataLoader:
    """Create training dataloader with pre-sharded dataset"""
    def worker_init_fn(worker_id):
        worker_seed = base_seed + rank * 1000 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # No need for distributed sampling since dataset is already sharded
    train_dataloader = DataLoader(
        dataset,  # pre-sharded dataset
        batch_size=args.batch_size,
        shuffle=True,  # Use shuffle directly instead of sampler
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(base_seed + rank * 10000)  # Control shuffle randomness
    )

    return train_dataloader


def create_test_dataloader(dataset, args: TrainingArguments,
                      data_collator, rank, world_size) -> tuple:
    """Create test dataloader with pre-sharded dataset"""
    # Test dataset requires DistributedSampler for distributed sampling
    sampler = DistributedSampler(
        dataset,
        shuffle=False,
        num_replicas=world_size,
        rank=rank
    )

    test_dataloader = DataLoader(
        dataset,    # non pre-sharded dataset
        batch_size=args.batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return test_dataloader
