from transformers import WhisperConfig


class WhisperVQConfig(WhisperConfig):
    def __init__(self,
                 pooling_kernel_size=None,
                 pooling_type="max",
                 pooling_position=0,
                 quantize_vocab_size=None,
                 quantize_position=16,
                 quantize_commit_coefficient=0.25,
                 quantize_loss_scale=1.0,
                 quantize_ema_decay=None,
                 quantize_restart_interval=None,
                 quantize_encoder_only=False,
                 quantize_causal_encoder=False,
                 quantize_causal_block_size=None,
                 skip_language_detection=False,
                 encoder_causal_attention=False,
                 encoder_causal_convolution=False,
                 encoder_residual_source_position=None,
                 encoder_residual_target_position=None,
                 use_residual_adapter=False,
                 use_residual_gate=False,
                 codebook_gaussian_init=False,
                 use_commit_loss=True,
                 **kwargs):
        """
        Initialize WhisperVQ configuration.

        Args:
            pooling_kernel_size (int, optional): Kernel size for pooling operation.
                Used for temporal downsampling of features.
            pooling_type (str): Type of pooling operation. Options: "max", "avg".
                Controls how temporal features are aggregated.
            pooling_position (int): Transformer layer index where pooling is applied.
                Determines at which depth temporal compression occurs.
            quantize_encoder_only (bool): If True, only quantizes encoder outputs.
                If False, quantizes the full model pipeline.
            quantize_vocab_size (int, optional): Size of the quantization codebook.
                Determines the number of discrete tokens available.
            quantize_position (int): Transformer layer index where quantization is applied.
                Controls where in the network discretization occurs.
            quantize_commit_coefficient (float): Coefficient for commitment loss.
                Balances between reconstruction and commitment to codebook entries.
            use_commit_loss (bool): Whether to apply commitment loss in quantization.
                Encourages encoder outputs to commit to codebook entries.
            **kwargs: Additional arguments passed to parent WhisperConfig.
        """
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        self.encoder_residual_source_position = encoder_residual_source_position
        self.encoder_residual_target_position = encoder_residual_target_position
        self.use_residual_adapter = use_residual_adapter
        self.use_residual_gate = use_residual_gate
        self.codebook_gaussian_init = codebook_gaussian_init
        self.use_commit_loss = use_commit_loss
        super().__init__(**kwargs)
