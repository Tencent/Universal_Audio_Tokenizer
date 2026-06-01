from dataclasses import dataclass


@dataclass
class TrainingArguments:
    """Training arguments for Whisper-VQ"""
    base_seed: int = 42
    batch_size: int = 4
    num_workers: int = 4
    num_train_epochs: int = 3
    num_train_steps: int = 1000000
    warmup_steps: int = 500
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    encoder_lr: float = None  # If None, use learning_rate
    decoder_lr: float = None  # If None, use learning_rate
    encoder_min_lr: float = None  # If None, use min_lr
    decoder_min_lr: float = None  # If None, use min_lr
    eval_begin: int = 0
    eval_steps: int = 1000
    eval_output_dir: str = "./eval_outputs"
    max_length: int = 225
    logging_steps: int = 1
    cache_dir: str = None
    model_output_dir: str = "./checkpoints"
    model_base_name: str = "openai/whisper-large-v3"
    model_type: str = "whisper"
    use_fsdp: bool = False
    max_grad_norm: float = 1.0
    freezed_encoder: bool = False
    freezed_decoder: bool = False
    use_ddp: bool = False
    is_debug_mode: bool = False
    scheduler_type: str = "onecycle"
    codebook_warmup: bool = False
    padding_strategy: str = "longest"
    save_steps: int = 10000000
    init_checkpoint: str = ''
    start_step: int = 0
    min_text_length: int = 0
    use_codebook_monitor: bool = True
    noise_contrastive: bool = False
    decoder_cold_start: bool = False

    @classmethod
    def from_dict(cls, config: dict):
        """Create instance from dictionary, using default values for missing keys"""
        # Use a dictionary comprehension to create the params dictionary, ensuring type conversion
        params = {}
        for k, v in cls.__annotations__.items():
            if k in config:
                # For learning_rate and related fields, ensure they are float types
                if k in ['learning_rate', 'min_lr', 'encoder_lr', 'decoder_lr', 'encoder_min_lr', 'decoder_min_lr'] \
                    and isinstance(config[k], str):
                    params[k] = float(config[k])
                else:
                    params[k] = v(config[k])  # Use the type for conversion
            else:
                params[k] = getattr(cls, k)  # Use default value
        return cls(**params)

    def __str__(self):
        """Define string representation"""
        params = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"TrainingArguments({', '.join(params)})"
