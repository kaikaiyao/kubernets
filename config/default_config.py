"""
Default configuration for StyleGAN watermarking.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union
import ast


@dataclass
class ModelConfig:
    """Model configuration."""
    stylegan2_url: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-ada.pkl"
    stylegan2_local_path: str = "ffhq70k-paper256-ada.pkl"
    img_size: int = 256
    key_length: int = 4
    selected_indices: Optional[List[int]] = None
    w_partial_set_seed: int = 42
    w_partial_length: int = 32
    use_image_pixels: bool = False
    image_pixel_set_seed: int = 42
    image_pixel_count: int = 8192
    key_mapper_seed: int = 2025
    key_mapper_use_sine: bool = False
    key_mapper_sensitivity: float = 20.0
    use_zca_whitening: bool = False
    zca_eps: float = 1e-5
    zca_batch_size: int = 1000
    direct_pixel_pred: bool = False
    # New: Mutual information estimation parameters
    estimate_mutual_info: bool = False
    mi_n_samples: int = 1000
    mi_k_neighbors: int = 3


@dataclass
class DecoderConfig:
    """Decoder configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048, 1024, 512, 256])
    activation: str = "gelu"
    dropout_rate: float = 0.3
    num_residual_blocks: int = 3
    use_spectral_norm: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    total_iterations: int = 100000
    lr: float = 1e-4
    lambda_lpips: float = 1.0
    log_interval: int = 1
    checkpoint_interval: int = 10000
    freeze_watermarked_model: bool = False
    direct_feature_decoder: bool = False


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    init_method: str = "env://"


@dataclass
class EvaluateConfig:
    """Configuration for evaluation."""
    batch_size: int = 16
    num_samples: int = 1000
    num_vis_samples: int = 10
    evaluation_mode: str = 'batch'  # Changed from 'both' to 'batch' as default
    
    # Process monitoring options
    enable_timing_logs: bool = True  # Enable detailed timing logs for process monitoring
    
    # Multi-decoder mode options
    enable_multi_decoder: bool = False  # Whether to enable multi-decoder mode
    multi_decoder_checkpoints: List[str] = field(default_factory=list)  # List of checkpoint paths for multi-decoder mode
    multi_decoder_key_lengths: List[int] = field(default_factory=list)  # List of key lengths for each decoder
    multi_decoder_key_mapper_seeds: List[int] = field(default_factory=list)  # List of key mapper seeds for each decoder
    multi_decoder_pixel_counts: List[int] = field(default_factory=list)  # List of pixel counts for each decoder
    multi_decoder_pixel_seeds: List[int] = field(default_factory=list)  # List of pixel seeds for each decoder
    
    # Visualization options
    enable_visualization: bool = False  # Master switch for all visualizations
    save_comparisons: bool = False  # Whether to save comparison visualizations
    visualization_seed: int = 42  # Random seed for consistent visualization samples
    verbose_visualization: bool = False  # Whether to log detailed per-sample information
    
    # Visualization flags for specific model types
    visualize_pretrained: bool = False  # Master switch for pretrained model visualizations
    visualize_transforms: bool = False  # Master switch for transformation visualizations
    visualize_ffhq1k: bool = False
    visualize_ffhq30k: bool = False
    visualize_ffhq70k_bcr: bool = False
    visualize_ffhq70k_noaug: bool = False
    
    # Visualization flags for specific transformations
    visualize_truncation: bool = False
    visualize_truncation_watermarked: bool = False
    visualize_quantization: bool = False
    visualize_quantization_watermarked: bool = False
    visualize_quantization_int4: bool = False
    visualize_quantization_int4_watermarked: bool = False
    visualize_quantization_int2: bool = False
    visualize_quantization_int2_watermarked: bool = False
    visualize_downsample: bool = False
    visualize_downsample_watermarked: bool = False
    visualize_jpeg: bool = False
    visualize_jpeg_watermarked: bool = False
    visualize_zca_whitening: bool = False
    visualize_zca_whitening_watermarked: bool = False
    
    # Negative sample evaluation options
    evaluate_neg_samples: bool = True
    
    # Pre-trained model options
    evaluate_pretrained: bool = True
    evaluate_ffhq1k: bool = True
    evaluate_ffhq30k: bool = True
    evaluate_ffhq70k_bcr: bool = True
    evaluate_ffhq70k_noaug: bool = True
    
    # Image transformation options
    evaluate_transforms: bool = True
    
    # Truncation options
    evaluate_truncation: bool = True
    evaluate_truncation_watermarked: bool = True
    truncation_psi: float = 2.0
    
    # Int8 quantization options (original)
    evaluate_quantization: bool = True
    evaluate_quantization_watermarked: bool = True
    
    # Int4 quantization options
    evaluate_quantization_int4: bool = True
    evaluate_quantization_int4_watermarked: bool = True
    
    # Int2 quantization options
    evaluate_quantization_int2: bool = True
    evaluate_quantization_int2_watermarked: bool = True
    
    # Downsampling options
    evaluate_downsample: bool = True
    evaluate_downsample_watermarked: bool = True
    downsample_size: int = 128
    
    # JPEG compression options
    evaluate_jpeg: bool = False
    evaluate_jpeg_watermarked: bool = False
    jpeg_quality: int = 55
    
    # ZCA whitening options
    evaluate_zca_whitening: bool = True
    evaluate_zca_whitening_watermarked: bool = True


@dataclass
class AttackConfig:
    """Configuration for attacks against the watermarking."""
    # General attack settings
    batch_size: int = 16
    num_samples: int = 100
    
    # PGD attack parameters
    pgd_alpha: float = 0.01  # Step size
    pgd_steps: int = 100     # Number of PGD iterations
    pgd_epsilon: float = 1.0  # Maximum perturbation
    
    # Surrogate training parameters
    surrogate_lr: float = 1e-4
    surrogate_batch_size: int = 32
    surrogate_epochs: int = 1
    surrogate_num_samples: int = 10000
    num_surrogate_models: int = 5
    use_combined_surrogate_input: bool = False  # Whether to use both images and w_partial as input to surrogate decoders
    
    # Attack evaluation
    log_interval: int = 10
    visualization_samples: int = 5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    output_dir: str = "results"
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = None
    
    def update_from_args(self, args, mode='train'):
        """Update config from command line arguments."""
        # Model configuration
        if hasattr(args, 'stylegan2_url'):
            self.model.stylegan2_url = args.stylegan2_url
        if hasattr(args, 'stylegan2_local_path'):
            self.model.stylegan2_local_path = args.stylegan2_local_path
        if hasattr(args, 'img_size'):
            self.model.img_size = args.img_size
        if hasattr(args, 'key_length'):
            self.model.key_length = args.key_length
        if hasattr(args, 'w_partial_set_seed'):
            self.model.w_partial_set_seed = args.w_partial_set_seed
        if hasattr(args, 'w_partial_length'):
            self.model.w_partial_length = args.w_partial_length
        if hasattr(args, 'use_image_pixels'):
            self.model.use_image_pixels = args.use_image_pixels
        if hasattr(args, 'image_pixel_set_seed'):
            self.model.image_pixel_set_seed = args.image_pixel_set_seed
        if hasattr(args, 'image_pixel_count'):
            self.model.image_pixel_count = args.image_pixel_count
        if hasattr(args, 'key_mapper_seed'):
            self.model.key_mapper_seed = args.key_mapper_seed
        if hasattr(args, 'key_mapper_use_sine'):
            self.model.key_mapper_use_sine = args.key_mapper_use_sine
        if hasattr(args, 'key_mapper_sensitivity'):
            self.model.key_mapper_sensitivity = args.key_mapper_sensitivity
        if hasattr(args, 'use_zca_whitening'):
            self.model.use_zca_whitening = args.use_zca_whitening
        if hasattr(args, 'zca_eps'):
            self.model.zca_eps = args.zca_eps
        if hasattr(args, 'zca_batch_size'):
            self.model.zca_batch_size = args.zca_batch_size
        if hasattr(args, 'direct_pixel_pred'):
            self.model.direct_pixel_pred = args.direct_pixel_pred
            if self.model.direct_pixel_pred:
                logging.info("Direct pixel prediction mode enabled")
        
        # New: Mutual information estimation parameters
        if hasattr(args, 'estimate_mutual_info'):
            self.model.estimate_mutual_info = args.estimate_mutual_info
        if hasattr(args, 'mi_n_samples'):
            self.model.mi_n_samples = args.mi_n_samples
        if hasattr(args, 'mi_k_neighbors'):
            self.model.mi_k_neighbors = args.mi_k_neighbors
        
        # Parse selected indices if provided
        if hasattr(args, 'selected_indices') and args.selected_indices:
            try:
                self.model.selected_indices = [int(idx) for idx in args.selected_indices.split(',')]
            except:
                logging.warning(f"Failed to parse selected_indices: {args.selected_indices}")
                self.model.selected_indices = None
        
        # Decoder configuration
        if hasattr(args, 'decoder_hidden_dims'):
            try:
                self.decoder.hidden_dims = [int(dim) for dim in args.decoder_hidden_dims.split(',')]
            except:
                logging.warning(f"Failed to parse decoder_hidden_dims: {args.decoder_hidden_dims}")
        
        if hasattr(args, 'decoder_activation'):
            self.decoder.activation = args.decoder_activation
        if hasattr(args, 'decoder_dropout_rate'):
            self.decoder.dropout_rate = args.decoder_dropout_rate
        if hasattr(args, 'decoder_num_residual_blocks'):
            self.decoder.num_residual_blocks = args.decoder_num_residual_blocks
        if hasattr(args, 'decoder_no_spectral_norm'):
            self.decoder.use_spectral_norm = not args.decoder_no_spectral_norm
        if hasattr(args, 'decoder_no_layer_norm'):
            self.decoder.use_layer_norm = not args.decoder_no_layer_norm
        if hasattr(args, 'decoder_no_attention'):
            self.decoder.use_attention = not args.decoder_no_attention
        
        # Mode-specific configuration
        if mode == 'train':
            if hasattr(args, 'batch_size'):
                self.training.batch_size = args.batch_size
            if hasattr(args, 'total_iterations'):
                self.training.total_iterations = args.total_iterations
            if hasattr(args, 'lr'):
                self.training.lr = args.lr
            if hasattr(args, 'lambda_lpips'):
                self.training.lambda_lpips = args.lambda_lpips
            if hasattr(args, 'log_interval'):
                self.training.log_interval = args.log_interval
            if hasattr(args, 'checkpoint_interval'):
                self.training.checkpoint_interval = args.checkpoint_interval
            if hasattr(args, 'freeze_watermarked_model'):
                self.training.freeze_watermarked_model = args.freeze_watermarked_model
            if hasattr(args, 'direct_feature_decoder'):
                self.training.direct_feature_decoder = args.direct_feature_decoder
        elif mode == 'evaluate':
            # Update evaluation-specific parameters
            if hasattr(args, 'evaluation_mode'):
                self.evaluate.evaluation_mode = args.evaluation_mode
            if hasattr(args, 'num_samples'):
                self.evaluate.num_samples = args.num_samples
            if hasattr(args, 'num_vis_samples'):
                self.evaluate.num_vis_samples = args.num_vis_samples
            if hasattr(args, 'batch_size'):
                self.evaluate.batch_size = args.batch_size
            if hasattr(args, 'evaluate_neg_samples'):
                self.evaluate.evaluate_neg_samples = args.evaluate_neg_samples
            if hasattr(args, 'evaluate_pretrained'):
                self.evaluate.evaluate_pretrained = args.evaluate_pretrained
            if hasattr(args, 'evaluate_transforms'):
                self.evaluate.evaluate_transforms = args.evaluate_transforms
            if hasattr(args, 'truncation_psi'):
                self.evaluate.truncation_psi = args.truncation_psi
            if hasattr(args, 'downsample_size'):
                self.evaluate.downsample_size = args.downsample_size
            if hasattr(args, 'jpeg_quality'):
                self.evaluate.jpeg_quality = args.jpeg_quality
            
            # ZCA whitening evaluation parameters
            if hasattr(args, 'evaluate_zca_whitening'):
                self.evaluate.evaluate_zca_whitening = args.evaluate_zca_whitening
            if hasattr(args, 'evaluate_zca_whitening_watermarked'):
                self.evaluate.evaluate_zca_whitening_watermarked = args.evaluate_zca_whitening_watermarked
            
            # Detailed evaluation options
            if hasattr(args, 'evaluate_ffhq1k'):
                self.evaluate.evaluate_ffhq1k = args.evaluate_ffhq1k
            if hasattr(args, 'evaluate_ffhq30k'):
                self.evaluate.evaluate_ffhq30k = args.evaluate_ffhq30k
            if hasattr(args, 'evaluate_ffhq70k_bcr'):
                self.evaluate.evaluate_ffhq70k_bcr = args.evaluate_ffhq70k_bcr
            if hasattr(args, 'evaluate_ffhq70k_noaug'):
                self.evaluate.evaluate_ffhq70k_noaug = args.evaluate_ffhq70k_noaug
            
            # Specific transformations
            if hasattr(args, 'evaluate_truncation'):
                self.evaluate.evaluate_truncation = args.evaluate_truncation
            if hasattr(args, 'evaluate_truncation_watermarked'):
                self.evaluate.evaluate_truncation_watermarked = args.evaluate_truncation_watermarked
            if hasattr(args, 'evaluate_quantization'):
                self.evaluate.evaluate_quantization = args.evaluate_quantization
            if hasattr(args, 'evaluate_quantization_watermarked'):
                self.evaluate.evaluate_quantization_watermarked = args.evaluate_quantization_watermarked
            if hasattr(args, 'evaluate_quantization_int4'):
                self.evaluate.evaluate_quantization_int4 = args.evaluate_quantization_int4
            if hasattr(args, 'evaluate_quantization_int4_watermarked'):
                self.evaluate.evaluate_quantization_int4_watermarked = args.evaluate_quantization_int4_watermarked
            if hasattr(args, 'evaluate_quantization_int2'):
                self.evaluate.evaluate_quantization_int2 = args.evaluate_quantization_int2
            if hasattr(args, 'evaluate_quantization_int2_watermarked'):
                self.evaluate.evaluate_quantization_int2_watermarked = args.evaluate_quantization_int2_watermarked
            if hasattr(args, 'evaluate_downsample'):
                self.evaluate.evaluate_downsample = args.evaluate_downsample
            if hasattr(args, 'evaluate_downsample_watermarked'):
                self.evaluate.evaluate_downsample_watermarked = args.evaluate_downsample_watermarked
            if hasattr(args, 'evaluate_jpeg'):
                self.evaluate.evaluate_jpeg = args.evaluate_jpeg
            if hasattr(args, 'evaluate_jpeg_watermarked'):
                self.evaluate.evaluate_jpeg_watermarked = args.evaluate_jpeg_watermarked
            
            # Multi-decoder mode parameters
            if hasattr(args, 'enable_multi_decoder'):
                self.evaluate.enable_multi_decoder = args.enable_multi_decoder
            if hasattr(args, 'multi_decoder_checkpoints'):
                self.evaluate.multi_decoder_checkpoints = args.multi_decoder_checkpoints
            if hasattr(args, 'multi_decoder_key_lengths'):
                self.evaluate.multi_decoder_key_lengths = args.multi_decoder_key_lengths
            if hasattr(args, 'multi_decoder_key_mapper_seeds'):
                self.evaluate.multi_decoder_key_mapper_seeds = args.multi_decoder_key_mapper_seeds
            if hasattr(args, 'multi_decoder_pixel_counts'):
                self.evaluate.multi_decoder_pixel_counts = args.multi_decoder_pixel_counts
            if hasattr(args, 'multi_decoder_pixel_seeds'):
                self.evaluate.multi_decoder_pixel_seeds = args.multi_decoder_pixel_seeds
                
            # Visualization parameters
            if hasattr(args, 'visualization_seed'):
                self.evaluate.visualization_seed = args.visualization_seed
            if hasattr(args, 'verbose_visualization'):
                self.evaluate.verbose_visualization = args.verbose_visualization
                
        elif mode == 'attack':
            # Update attack-specific parameters
            if hasattr(args, 'batch_size'):
                self.attack.batch_size = args.batch_size
            if hasattr(args, 'num_samples'):
                self.attack.num_samples = args.num_samples
            if hasattr(args, 'pgd_alpha'):
                self.attack.pgd_alpha = args.pgd_alpha
            if hasattr(args, 'pgd_steps'):
                self.attack.pgd_steps = args.pgd_steps
            if hasattr(args, 'pgd_epsilon'):
                self.attack.pgd_epsilon = args.pgd_epsilon
            if hasattr(args, 'surrogate_lr'):
                self.attack.surrogate_lr = args.surrogate_lr
            if hasattr(args, 'surrogate_batch_size'):
                self.attack.surrogate_batch_size = args.surrogate_batch_size
            if hasattr(args, 'surrogate_epochs'):
                self.attack.surrogate_epochs = args.surrogate_epochs
            if hasattr(args, 'surrogate_num_samples'):
                self.attack.surrogate_num_samples = args.surrogate_num_samples
            if hasattr(args, 'num_surrogate_models'):
                self.attack.num_surrogate_models = args.num_surrogate_models
            if hasattr(args, 'use_combined_surrogate_input'):
                self.attack.use_combined_surrogate_input = args.use_combined_surrogate_input
            if hasattr(args, 'log_interval'):
                self.attack.log_interval = args.log_interval
            if hasattr(args, 'visualization_samples'):
                self.attack.visualization_samples = args.visualization_samples
        
        # Other configuration
        if hasattr(args, 'output_dir'):
            self.output_dir = args.output_dir
        if hasattr(args, 'checkpoint_path'):
            self.checkpoint_path = args.checkpoint_path
        if hasattr(args, 'seed'):
            self.seed = args.seed


def get_default_config() -> Config:
    """Get default configuration."""
    return Config() 