import argparse
from datetime import datetime, timedelta
import logging
import os
import random
import yaml
import numpy as np
from safetensors.torch import load_file, save_model
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import wandb
from src.model.configuration_whisper import WhisperVQConfig
from src.model.modeling_whisper import WhisperVQForConditionalGeneration
from src.train.custom_collator import CustomDataCollatorSpeechSeq2SeqWithPadding
from src.train.custom_dataloader import create_train_dataloader, create_test_dataloader
from src.train.custom_dataset import prepare_custom_dataset
from src.train.metrics import (
    evaluate_asr,
    evaluate_ser,
    evaluate_cap,
    evaluate_cap_only_emotion,
    evaluate_sap,
    evaluate_sap_only_emotion,
)
from src.train.training_args import TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--config_file", type=str, required=True, help="config file")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_config(config_file):
    """Load and return configuration from yaml file."""
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise


def init_wandb(wandb_config):
    """Initialize weights and biases logging."""
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.login(key=wandb_config["key"], host=wandb_config["host"])
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["team"],
        name=wandb_config["name"] + "_" + time_stamp,
        group=wandb_config["group"],
        dir=wandb_config["dir"],
    )


def init_model_and_tokenizer(
    model_name,
    model_type,
    model_config_args,
    feature_extraction_config_args,
    training_args
):
    """Initialize model, tokenizer, and feature extractor."""
    logging.info(f"Loading `{model_type}` model from {model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, **feature_extraction_config_args)
    if dist.get_rank() == 0:
        logging.info("Feature Extractor Configurations:")
        logging.info(f"  {feature_extractor.sampling_rate = }")
        logging.info(f"  {feature_extractor.feature_size = }")
        logging.info(f"  {feature_extractor.chunk_length = }")
        logging.info(f"  {feature_extractor.n_samples = }")
        logging.info(f"  {feature_extractor.nb_max_frames = }")
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    # Main process prints model configuration parameters
    if dist.get_rank() == 0:
        logging.info("Model configuration arguments:")
        for key, value in model_config_args.items():
            logging.info(f"  {key}: {value}")

    model_config = WhisperVQConfig.from_pretrained(
        model_name,
        **model_config_args
    )

    model = WhisperVQForConditionalGeneration.from_pretrained(
        model_name,
        config=model_config,
        ignore_mismatched_sizes=True  # Allows size mismatch, handles automatically
    )

    # Add special tokens
    existing_special_tokens = tokenizer.special_tokens_map["additional_special_tokens"]
    tokenizer.add_special_tokens({
        "additional_special_tokens": existing_special_tokens + ["<|emotion|>", "<|startofcaption|>", "<|startofsap|>"]
    })
    model.resize_token_embeddings(len(tokenizer))

    # Decoder cold start - initialize only in main process
    if training_args.decoder_cold_start:
        if dist.get_rank() == 0:
            logging.info("Cold starting decoder - reinitializing decoder weights")

            # Re-initialize all parameters of the decoder
            def reinitialize_module(module):
                """Recursively re-initialize module parameters"""
                for name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        if 'weight' in name:
                            if len(param.shape) >= 2:
                                # Initialize using Xavier
                                torch.nn.init.xavier_uniform_(param)
                            else:
                                # Use normal distribution for 1D parameters
                                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                        elif 'bias' in name:
                            torch.nn.init.zeros_(param)

                # Recursively process submodules
                for child in module.children():
                    reinitialize_module(child)

            # Re-initialize decoder
            if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                reinitialize_module(model.model.decoder)
                logging.info("Decoder weights reinitialized")

            # Re-initialize lm_head (language model head)
            if hasattr(model, 'proj_out'):
                torch.nn.init.xavier_uniform_(model.proj_out.weight)
                if model.proj_out.bias is not None:
                    torch.nn.init.zeros_(model.proj_out.bias)
                logging.info("Language model head (proj_out) reinitialized")

        # Wait for main process to complete initialization
        dist.barrier()

    # Update checkpoint loading logic to use safetensors - load only in main process
    if training_args.init_checkpoint:
        if dist.get_rank() == 0:
            logging.info(f"Loading checkpoint from {training_args.init_checkpoint}")
            checkpoint = load_file(training_args.init_checkpoint)
            model.load_state_dict(checkpoint, strict=False)
            logging.info("Checkpoint loaded successfully")

        # Wait for main process to complete checkpoint loading
        dist.barrier()

    return model, tokenizer, feature_extractor


def setup_optimizer_and_scheduler(
    model,
    num_training_steps,
    learning_rate,
    min_lr,
    warmup_steps,
    scheduler_type="onecycle",
    encoder_lr=None,
    decoder_lr=None,
    encoder_min_lr=None,
    decoder_min_lr=None
):
    """Setup optimizer and learning rate scheduler

    Args:
        model: The model to optimize
        num_training_steps: Total number of training steps
        learning_rate: Default learning rate (used for all params if encoder_lr/decoder_lr not specified)
        min_lr: Default minimum learning rate
        warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler ("onecycle", "constant", "cosine")
        encoder_lr: Learning rate for encoder parameters (if None, uses learning_rate)
        decoder_lr: Learning rate for decoder parameters (if None, uses learning_rate)
        encoder_min_lr: Minimum learning rate for encoder (if None, uses min_lr)
        decoder_min_lr: Minimum learning rate for decoder (if None, uses min_lr)
    """
    # Determine if we need separate learning rates
    use_separate_lr = (encoder_lr is not None) or (decoder_lr is not None)

    if use_separate_lr:
        # Use separate learning rates for encoder and decoder
        actual_encoder_lr = encoder_lr if encoder_lr is not None else learning_rate
        actual_decoder_lr = decoder_lr if decoder_lr is not None else learning_rate
        actual_encoder_min_lr = encoder_min_lr if encoder_min_lr is not None else min_lr
        actual_decoder_min_lr = decoder_min_lr if decoder_min_lr is not None else min_lr

        # Get the actual model (unwrap if wrapped in DDP/DataParallel)
        actual_model = model.module if hasattr(model, 'module') else model

        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []
        other_params = []

        for name, param in actual_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name or 'proj_out' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': actual_encoder_lr,
                'name': 'encoder'
            })
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': actual_decoder_lr,
                'name': 'decoder'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': learning_rate,
                'name': 'other'
            })

        optimizer = torch.optim.AdamW(param_groups)

        # Log parameter group information
        if dist.get_rank() == 0:
            logging.info(f"Using separate learning rates:")
            logging.info(f"  Encoder LR: {actual_encoder_lr:.2e} -> {actual_encoder_min_lr:.2e}")
            logging.info(f"  Decoder LR: {actual_decoder_lr:.2e} -> {actual_decoder_min_lr:.2e}")
            logging.info(f"  Encoder params: {len(encoder_params)}")
            logging.info(f"  Decoder params: {len(decoder_params)}")
            logging.info(f"  Other params: {len(other_params)}")
    else:
        # Use single learning rate for all parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        actual_encoder_lr = learning_rate
        actual_decoder_lr = learning_rate
        actual_encoder_min_lr = min_lr
        actual_decoder_min_lr = min_lr

    if scheduler_type == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[actual_encoder_lr, actual_decoder_lr, learning_rate] if use_separate_lr else learning_rate,
            total_steps=num_training_steps,
            pct_start=warmup_steps/num_training_steps
        )
    elif scheduler_type == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_training_steps
        )
    elif scheduler_type == "cosine":
        # Cosine annealing with warmup
        if use_separate_lr:
            # For separate LRs, we need to create custom lambda functions for each parameter group
            def create_lambda_lr(group_idx, initial_lr, min_lr_val):
                def lr_lambda(step):
                    if step < warmup_steps:
                        # Warmup phase
                        return (1e-6 + (initial_lr - 1e-6) * step / warmup_steps) / initial_lr
                    elif step < num_training_steps:
                        # Cosine decay phase
                        progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                        return (min_lr_val + (initial_lr - min_lr_val) * cosine_decay) / initial_lr
                    else:
                        # Constant phase at min_lr
                        return min_lr_val / initial_lr
                return lr_lambda

            # Create lambda functions for each parameter group
            lambda_funcs = []
            param_group_lrs = []
            param_group_min_lrs = []

            for i, group in enumerate(optimizer.param_groups):
                if group['name'] == 'encoder':
                    param_group_lrs.append(actual_encoder_lr)
                    param_group_min_lrs.append(actual_encoder_min_lr)
                elif group['name'] == 'decoder':
                    param_group_lrs.append(actual_decoder_lr)
                    param_group_min_lrs.append(actual_decoder_min_lr)
                else:
                    param_group_lrs.append(learning_rate)
                    param_group_min_lrs.append(min_lr)

                lambda_funcs.append(create_lambda_lr(i, param_group_lrs[i], param_group_min_lrs[i]))

            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_funcs)
        else:
            # Original single LR implementation
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6 / learning_rate,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=min_lr
            )
            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=float('inf')
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
                milestones=[warmup_steps, num_training_steps]
            )
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    return optimizer, lr_scheduler


def compute_gradient_stats_memory_efficient(params):
    """
    Memory-efficient gradient stats computation, using online algorithm to avoid tensor concat
    """
    if not params:
        return {}

    # Calculate statistics using online algorithm
    total_norm_squared = 0.0
    total_elements = 0
    sum_values = 0.0
    sum_squared_values = 0.0
    global_min = float('inf')
    global_max = float('-inf')

    for param in params:
        if param.grad is not None and param.requires_grad:
            grad = param.grad.data

            # Calculate statistics for current parameter
            param_norm_squared = torch.sum(grad * grad).item()
            param_sum = torch.sum(grad).item()
            param_sum_squared = torch.sum(grad * grad).item()
            param_min = torch.min(grad).item()
            param_max = torch.max(grad).item()
            param_numel = grad.numel()

            # Accumulate statistics
            total_norm_squared += param_norm_squared
            total_elements += param_numel
            sum_values += param_sum
            sum_squared_values += param_sum_squared
            global_min = min(global_min, param_min)
            global_max = max(global_max, param_max)

    if total_elements == 0:
        return {}

    # Calculate final statistics
    global_norm = total_norm_squared ** 0.5
    global_mean = sum_values / total_elements
    global_var = (sum_squared_values / total_elements) - (global_mean ** 2)
    global_std = max(0.0, global_var) ** 0.5  # Ensure non-negative

    return {
        'global_grad_norm': global_norm,
        'global_grad_mean': global_mean,
        'global_grad_std': global_std,
        'global_grad_max': global_max,
        'global_grad_min': global_min,
    }


def train_step(batch, model, optimizer, scaler, device, max_grad_norm=1.0):
    """Execute single training step"""
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        outputs = model(**batch)
        loss = outputs.loss
        quantized_loss = outputs.quantized_loss
        num_update_indices = outputs.num_update_indices
        codebook_usage = outputs.codebook_usage
        ce_loss = outputs.ce_loss
        residual_gamma_mean = outputs.residual_gamma_mean
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Get parameter list (to avoid generator reuse issues)
    if hasattr(model, 'module'):
        model_params = list(model.module.parameters())
    else:
        model_params = list(model.parameters())

    # Calculate stats before gradient clipping (memory-efficient version)
    grad_stats_before = compute_gradient_stats_memory_efficient(model_params)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Calculate stats after gradient clipping (memory-efficient version)
    grad_stats_after = compute_gradient_stats_memory_efficient(model_params)

    scaler.step(optimizer)
    scaler.update()

    quantized_loss = 0 if quantized_loss is None else quantized_loss.item()
    num_update_indices = 0 if num_update_indices is None else num_update_indices
    residual_gamma_mean = 0 if residual_gamma_mean is None else residual_gamma_mean.item()

    return (
        loss.item(),
        quantized_loss,
        ce_loss.item(),
        num_update_indices,
        codebook_usage,
        grad_stats_before,
        grad_stats_after,
        residual_gamma_mean,
    )


def create_weighted_dataset(config_dict, global_rank, world_size, training_args, is_main_process):
    """Create weighted dataset"""
    datasets_with_weights = []
    combined_train_dataset_name = ''

    for key, value in config_dict["data_args"]["train_data_file"].items():
        if is_main_process:
            logging.info(f"Preparing training dataset `{key}`...")

        # Support two configuration formats
        if isinstance(value, dict) and 'path' in value:
            data_path = value['path']
            weight = value.get('weight', 1)  # Default weight is 1
            # Ensure weight is integer
            if not isinstance(weight, int) or weight < 1:
                raise ValueError(f"Dataset '{key}' weight must be a positive integer, got: {weight}")
        else:
            # Compatible with original format
            data_path = value
            weight = 1

        train_dataset = prepare_custom_dataset(
            data_path,
            rank=global_rank,
            world_size=world_size,
            cache_dir=os.path.join(training_args.cache_dir, key),
            debug_mode=training_args.is_debug_mode,
            is_test_data=False
        )

        if train_dataset is not None:
            original_size = len(train_dataset)
            logging.info(f"[rank {global_rank}] `{key}` original data size: {original_size}, weight: {weight}")

            # Simply repeat dataset based on integer weights
            if weight > 1:
                logging.info(f'before repeat for `{key}`: {len(train_dataset)}')
                repeated_datasets = [train_dataset] * weight
                train_dataset = torch.utils.data.ConcatDataset(repeated_datasets)
                logging.info(f'after repeat for `{key}`: {len(train_dataset)}')

            datasets_with_weights.append(train_dataset)
            combined_train_dataset_name += f",{key}" if combined_train_dataset_name else key

            # Record final dataset size
            final_size = len(train_dataset)
            if is_main_process:
                wandb.log({
                    f"dataset/{key}_original_size": original_size,
                    f"dataset/{key}_final_size": final_size,
                    f"dataset/{key}_weight": weight
                })
                logging.info(f"Final dataset `{key}` size after {weight}x upsampling: {final_size}")

    # Concatenate all datasets
    if datasets_with_weights:
        combined_train_dataset = torch.utils.data.ConcatDataset(datasets_with_weights)
    else:
        combined_train_dataset = None

    return combined_train_dataset, combined_train_dataset_name


def train():
    args = parse_args()
    config_dict = load_config(args.config_file)
    training_args = TrainingArguments.from_dict(config_dict["training_args"])

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    base_seed = training_args.base_seed
    set_seed(base_seed + global_rank)  # Each process gets a different seed based on its global rank

    logging.info(f"global rank: {global_rank}, local rank: {local_rank}, world size: {world_size}")

    is_main_process = global_rank == 0

    # Set device for the current process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize wandb for main process
    if is_main_process:
        logging.info(f"Initializing wandb")
        wandb_args = config_dict.get("wandb_args", {})
        logging.info(f"wandb args: {wandb_args}")
        init_wandb(wandb_args)

    # Initialize model and tokenizer
    model_name = training_args.model_base_name
    model_type = training_args.model_type
    model, tokenizer, feature_extractor = init_model_and_tokenizer(
        model_name,
        model_type,
        config_dict["model_args"],
        config_dict["feature_extraction_args"] if "feature_extraction_args" in config_dict else {},
        training_args
    )

    if training_args.freezed_encoder:
        model.freeze_encoder()
    if training_args.freezed_decoder:
        model.freeze_decoder()

    # 3. Prepare datasets
    combined_train_dataset, combined_train_dataset_name = create_weighted_dataset(
        config_dict, global_rank, world_size, training_args, is_main_process
    )

    data_size = torch.tensor(len(combined_train_dataset), dtype=torch.int64, device=device)
    logging.info(f"[rank {global_rank}] train dataset size: {data_size.item()}")
    dist.all_reduce(data_size, op=dist.ReduceOp.SUM)
    if is_main_process:
        wandb.log({f"dataset/total_train_dataset_size": data_size.item()})
        logging.info(f"Total train dataset size: {data_size.item()}")

    train_data_collator = CustomDataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        padding_strategy=training_args.padding_strategy,
        model_config=model.config,
        dataset_name=combined_train_dataset_name,
        min_text_length=training_args.min_text_length if training_args.min_text_length is not None else 0,
    )

    train_dataloader = create_train_dataloader(
        combined_train_dataset, training_args, train_data_collator, base_seed=base_seed, rank=global_rank
    )

    training_args.train_dataset_size = len(train_dataloader.dataset)

    test_datasets_by_task = {}

    for task_type, task_datasets in config_dict["data_args"]["test_data_files"].items():
        test_datasets_by_task[task_type] = []
        for key, value in task_datasets.items():
            if is_main_process:
                logging.info(f"Preparing {task_type} test dataset `{key}`...")
            # Set `cache_dir=None` to eliminate potential read/write problems in multi-process environments.
            dataset = prepare_custom_dataset(value, rank=global_rank, world_size=world_size, cache_dir=None, debug_mode=training_args.is_debug_mode, is_test_data=True)
            if global_rank == 0:
                logging.info(f'{task_type} dataset `{key}` size: {len(dataset)}')
            test_datasets_by_task[task_type].append((key, dataset))

    # Create corresponding dataloader for each eval task
    eval_dataloaders_by_task = {}

    for task_type, dataset_list in test_datasets_by_task.items():
        eval_dataloaders_by_task[task_type] = []

        data_collator_class = CustomDataCollatorSpeechSeq2SeqWithPadding
        for key, dataset in dataset_list:
            data_collator = data_collator_class(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                decoder_start_token_id=model.config.decoder_start_token_id,
                padding_strategy=training_args.padding_strategy,
                model_config=model.config,
                dataset_name=key,
            )
            eval_dataloader = create_test_dataloader(
                dataset, training_args, data_collator, global_rank, world_size
            )
            eval_dataloaders_by_task[task_type].append((key, eval_dataloader))


    num_training_steps = training_args.num_train_steps
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(
        model,
        num_training_steps,
        training_args.learning_rate,
        training_args.min_lr,
        training_args.warmup_steps,
        training_args.scheduler_type,
        encoder_lr=training_args.encoder_lr,
        decoder_lr=training_args.decoder_lr,
        encoder_min_lr=training_args.encoder_min_lr,
        decoder_min_lr=training_args.decoder_min_lr
    )
    model = model.to(device)

    # Fix NaN in codebook and ema_weight (execute only on rank 0)
    if dist.get_rank() == 0:
        encoder = model.model.encoder if hasattr(model.model, 'encoder') else model.get_encoder()
        if hasattr(encoder, 'codebook') and encoder.codebook is not None:
            if torch.isnan(encoder.codebook.weight).any() or torch.isinf(encoder.codebook.weight).any():
                logging.info("Detected NaN/Inf in codebook, reinitializing...")
                d_model = encoder.codebook.weight.shape[1]
                if hasattr(encoder.config, 'codebook_gaussian_init') and encoder.config.codebook_gaussian_init:
                    encoder.codebook.weight.data.copy_(
                        torch.randn_like(encoder.codebook.weight) * d_model ** -0.5
                    )
                else:
                    nn.init.normal_(encoder.codebook.weight)

            if hasattr(encoder, 'ema_weight') and encoder.ema_weight is not None:
                if torch.isnan(encoder.ema_weight).any() or torch.isinf(encoder.ema_weight).any():
                    logging.info("Detected NaN/Inf in ema_weight, reinitializing from codebook...")
                    encoder.ema_weight.copy_(encoder.codebook.weight.data.clone().float())

    # Broadcast model weights from main process to all processes (after moving to GPU)
    if is_main_process:
        logging.info("Broadcasting model weights from rank 0 to all processes...")

    for name, param in model.named_parameters():
        dist.broadcast(param.data, src=0)

    # Sync all buffers
    for name, buffer in model.named_buffers():
        dist.broadcast(buffer.data, src=0)

    dist.barrier()

    if is_main_process:
        logging.info("Model weights and buffers broadcasting completed")

    scaler = torch.amp.GradScaler('cuda')

    if training_args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model)

    model.train()

    if is_main_process:
        logging.info("Training started")

    global_step = 0
    start_epoch = 0
    steps_per_epoch = len(train_dataloader)

    if is_main_process:
        logging.info(f"Steps per epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, training_args.num_train_epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch}",
                         total=steps_per_epoch, disable=not is_main_process)):

            (
                loss,
                quantized_loss,
                ce_loss,
                num_update_indices,
                codebook_usage,
                grad_stats_before,
                grad_stats_after,
                residual_gamma_mean
            ) = train_step(batch, model, optimizer, scaler, device)

            lr_scheduler.step()
            global_step += 1

            if is_main_process and global_step % training_args.logging_steps == 0:
                # Process gradient statistics
                wandb_grad_stats = {}

                # Statistics before gradient clipping
                if grad_stats_before:
                    wandb_grad_stats.update({
                        'train/gradient/before_clip/global_grad_norm': grad_stats_before['global_grad_norm'],
                        'train/gradient/before_clip/global_grad_mean': grad_stats_before['global_grad_mean'],
                        'train/gradient/before_clip/global_grad_std': grad_stats_before['global_grad_std'],
                    })

                # Statistics after gradient clipping
                if grad_stats_after:
                    wandb_grad_stats.update({
                        'train/gradient/after_clip/global_grad_norm': grad_stats_after['global_grad_norm'],
                        'train/gradient/after_clip/global_grad_mean': grad_stats_after['global_grad_mean'],
                        'train/gradient/after_clip/global_grad_std': grad_stats_after['global_grad_std'],
                    })

                if grad_stats_before:
                    grad_norm_before = grad_stats_before['global_grad_norm']
                    if grad_norm_before > 10.0:
                        logging.warning(f"Step {global_step}: Gradient explosion detected, gradient norm before clipping: {grad_norm_before:.6f}")
                    elif grad_norm_before < 1e-7:
                        logging.warning(f"Step {global_step}: Vanishing gradient detected, gradient norm before clipping: {grad_norm_before:.6f}")

                # Main training logs
                wandb_logs = {
                    "train/loss": loss,
                    "train/ce_loss": ce_loss,
                    "train/quantized_loss": quantized_loss,
                    "train/num_update_indices": num_update_indices,
                    "train/codebook_usage": codebook_usage,
                    "train/epoch": epoch,
                    "train/residual_gamma_mean": residual_gamma_mean,
                    "step": global_step,
                }

                # Log learning rates (handle both single and multiple learning rates)
                last_lrs = lr_scheduler.get_last_lr()
                if len(last_lrs) == 1:
                    wandb_logs["train/lr"] = last_lrs[0]
                else:
                    # Multiple learning rates (encoder, decoder, other)
                    for i, lr_val in enumerate(last_lrs):
                        group_name = optimizer.param_groups[i].get('name', f'group_{i}')
                        wandb_logs[f"train/lr_{group_name}"] = lr_val
                    wandb_logs["train/lr"] = last_lrs[0]  # Also log first LR for compatibility

                # Merge gradient statistics
                wandb_logs.update(wandb_grad_stats)

                wandb.log(wandb_logs)

            if is_main_process:
                logging.info(f"Step {global_step}: loss = {loss:.4f}")
            dist.barrier()

            if (global_step % training_args.save_steps == 0) and is_main_process:
                save_dir = f"{training_args.model_output_dir}/checkpoint-epoch-{epoch}-step-{global_step}"
                os.makedirs(save_dir, exist_ok=True)
                save_model(model.module, f"{save_dir}/model.safetensors")
                model.module.config.save_pretrained(f"{save_dir}")
                feature_extractor.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

            if (global_step % training_args.eval_steps == 0) and (global_step >= training_args.eval_begin):
                output_dir = None
                if is_main_process:
                    output_dir = f"{training_args.eval_output_dir}/eval-epoch-{epoch}-step-{global_step}"
                    os.makedirs(output_dir, exist_ok=True)
                dist.barrier()

                with torch.no_grad():
                    # Evaluate all tasks
                    for task_type, eval_dataloader_list in eval_dataloaders_by_task.items():
                        for key, eval_dataloader in eval_dataloader_list:
                            if task_type == 'asr':
                                # ASR task uses WER for evaluation
                                avg_wer = evaluate_asr(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_wer.json" if is_main_process else None
                                )
                                if is_main_process:
                                    # logging.info(f'global_step at eval asr: {global_step}')
                                    wandb.log({f"eval/{task_type}/{key}_wer": avg_wer, "step": global_step})
                                    logging.info(f'{task_type}/{key} Average WER: {avg_wer:.2f}')

                            elif task_type == 'ser':
                                # Emotion task uses accuracy for evaluation
                                avg_accuracy = evaluate_ser(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_emotion_results.json" if is_main_process else None
                                )
                                if is_main_process:
                                    wandb.log({f"eval/{task_type}/{key}_accuracy": avg_accuracy, "step": global_step})
                                    logging.info(f'{task_type}/{key} Average Accuracy: {avg_accuracy:.2f}')

                            elif task_type == 'caption':
                                logging.info(f"========== [RANK {global_rank}] Evaluating caption task for dataset `{key}` ==========")
                                results = evaluate_cap(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_caption_results.json" if is_main_process else None
                                )
                                if is_main_process:
                                    wandb.log({
                                        f"eval/{task_type}/{key}_overall_score": results['overall_consistency_score'],
                                        f"eval/{task_type}/{key}_valid_json": results['valid_json_rate'],
                                        f"eval/{task_type}/{key}_wer": results['wer_stats']['wer'],
                                        f"eval/{task_type}/{key}_age_acc": results['age_accuracy'],
                                        f"eval/{task_type}/{key}_gender_acc": results['gender_accuracy'],
                                        f"eval/{task_type}/{key}_emotion_acc": results['emotion_accuracy'],
                                        f"eval/{task_type}/{key}_accent_acc": results['accent_accuracy'],
                                        f"eval/{task_type}/{key}_prosody_acc": results['prosody_accuracy'],
                                        f"eval/{task_type}/{key}_timbre_acc": results['timbre_accuracy'],
                                        "step": global_step,
                                    })
                                    logging.info(f"{task_type}/{key} Overall Consistency Score: {results['overall_consistency_score']:.4f}")
                                    logging.info(f"{task_type}/{key} JSON valid rate: {results['valid_json_rate']:.4f}")
                                    logging.info(f"{task_type}/{key} WER: {results['wer_stats']['wer']:.4f}")
                                    logging.info(f"{task_type}/{key} Age Accuracy: {results['age_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Gender Accuracy: {results['gender_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Emotion Accuracy: {results['emotion_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Accent Accuracy: {results['accent_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Prosody Accuracy: {results['prosody_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Timbre Accuracy: {results['timbre_accuracy']:.4f}")

                            elif task_type == 'caption_emotion':
                                results = evaluate_cap_only_emotion(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_caption_emotion_results.json" if is_main_process else None
                                )
                                if is_main_process:
                                    wandb.log({
                                        f"eval/{task_type}/{key}_valid_json": results['valid_json_rate'],
                                        f"eval/{task_type}/{key}_emotion_acc": results['emotion_accuracy'],
                                        "step": global_step,
                                    })
                                    logging.info(f"{task_type}/{key} JSON valid rate: {results['valid_json_rate']:.4f}")
                                    logging.info(f"{task_type}/{key} Emotion Accuracy: {results['emotion_accuracy']:.4f}")

                            elif task_type == 'sap':
                                # SAP task evaluates JSON validity rate
                                results = evaluate_sap(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_sap_results.json" if is_main_process else None
                                )
                                if is_main_process:
                                    wandb.log({
                                        # f"eval/{task_type}/{key}_overall_score": results['overall_consistency_score'],
                                        f"eval/{task_type}/{key}_valid_json": results['valid_json_rate'],
                                        f"eval/{task_type}/{key}_wer": results['wer_stats']['wer'],
                                        f"eval/{task_type}/{key}_age_acc": results['age_accuracy'],
                                        f"eval/{task_type}/{key}_gender_acc": results['gender_accuracy'],
                                        f"eval/{task_type}/{key}_emotion_acc": results['emotion_accuracy'],
                                        f"eval/{task_type}/{key}_accent_acc": results['accent_accuracy'],
                                        f"eval/{task_type}/{key}_prosody_acc": results['prosody_accuracy'],
                                        f"eval/{task_type}/{key}_timbre_acc": results['timbre_accuracy'],
                                        f"eval/{task_type}/{key}_non_linguistic_overall_score": results['non_linguistic_overall_consistency_score'],
                                        "step": global_step,
                                    })
                                    logging.info(f"{task_type}/{key} JSON valid rate: {results['valid_json_rate']:.4f}")
                                    logging.info(f"{task_type}/{key} WER: {results['wer_stats']['wer']:.4f}")
                                    logging.info(f"{task_type}/{key} Age Accuracy: {results['age_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Gender Accuracy: {results['gender_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Emotion Accuracy: {results['emotion_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Accent Accuracy: {results['accent_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Prosody Accuracy: {results['prosody_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Timbre Accuracy: {results['timbre_accuracy']:.4f}")
                                    logging.info(f"{task_type}/{key} Non-Linguistic Overall Consistency Score: {results['non_linguistic_overall_consistency_score']:.4f}")

                            elif task_type == 'sap-emotion':
                                results = evaluate_sap_only_emotion(
                                    model, eval_dataloader, tokenizer, device,
                                    max_length=training_args.max_length, rank=global_rank,
                                    output_file=f"{output_dir}/{key}_sap_emotion_results.json" if is_main_process else None
                                )
                                if is_main_process:
                                    wandb.log({
                                        f"eval/{task_type}/{key}_valid_json": results['valid_json_rate'],
                                        f"eval/{task_type}/{key}_emotion_acc": results['emotion_accuracy'],
                                        "step": global_step,
                                    })
                                    logging.info(f"{task_type}/{key} JSON valid rate: {results['valid_json_rate']:.4f}")
                                    logging.info(f"{task_type}/{key} Emotion Accuracy: {results['emotion_accuracy']:.4f}")

                dist.barrier()
                model.train()  # Set model back to training mode

            if global_step >= training_args.num_train_steps:
                break

        # Save at the end of each epoch
        if is_main_process:
            save_dir = f"{training_args.model_output_dir}/checkpoint-epoch-{epoch}-final"
            os.makedirs(save_dir, exist_ok=True)
            save_model(model.module, f"{save_dir}/model.safetensors")
            model.module.config.save_pretrained(f"{save_dir}")
            feature_extractor.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # Final model saving
    if is_main_process:
        final_save_dir = f"{training_args.model_output_dir}/checkpoint-final"
        os.makedirs(final_save_dir, exist_ok=True)
        save_model(model.module, f"{final_save_dir}/model.safetensors")
        model.module.config.save_pretrained(final_save_dir)
        feature_extractor.save_pretrained(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)

    if is_main_process:
        logging.info("Training completed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize distributed training
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    train()
    dist.destroy_process_group()
