import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20

from key.key_utils import flip_key

class CryptoCNN(nn.Module):
    """A convolutional neural network with cryptographic weight initialization.
    
    Uses ChaCha20 encryption for secure parameter initialization with fan-in aware
    scaling for improved convergence properties.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        binary_key: 256-bit secret key for parameter initialization
    """

    def __init__(self, input_channels: int, output_channels: int, binary_key: bytes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self._init_layers_with_scaling(binary_key)

    def _init_layers_with_scaling(self, binary_key: bytes) -> None:
        """Initializes network parameters using encrypted seed with fan-in scaling.
        
        Args:
            binary_key: 256-bit secret key for ChaCha20 encryption
        """
        # Generate cryptographic seed
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        seed_size = total_params * 4  # 4 bytes per float32
        rng = np.random.default_rng(seed=0)
        seed = rng.bytes(seed_size)
        
        # Encrypt seed using ChaCha20
        chacha_key = binary_key[:32]  # 256-bit key
        nonce = b'\x00' * 8  # Fixed nonce (cryptographically insecure for actual deployment, used only for reproducibility)
        cipher = ChaCha20.new(key=chacha_key, nonce=nonce)
        ciphertext = cipher.encrypt(seed)
        
        # Convert to normalized float32 array
        uint32_vals = np.frombuffer(ciphertext, dtype='>u4')
        normalized_floats = uint32_vals.astype(np.float32) / np.float32(0xFFFFFFFF)
        
        # Parameter initialization with fan-in scaling
        pointer = 0
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            bound = 1.0 / np.sqrt(fan_in)
            
            # Weight initialization
            weight_size = np.prod(layer.weight.shape)
            layer_weights = normalized_floats[pointer:pointer + weight_size]
            scaled_weights = (layer_weights * 2 * bound) - bound
            layer.weight.data = torch.from_numpy(scaled_weights.reshape(layer.weight.shape)).float()
            pointer += weight_size
            
            # Bias initialization
            if layer.bias is not None:
                bias_size = np.prod(layer.bias.shape)
                layer_biases = normalized_floats[pointer:pointer + bias_size]
                scaled_biases = (layer_biases * 2 * bound) - bound
                layer.bias.data = torch.from_numpy(scaled_biases.reshape(layer.bias.shape)).float()
                pointer += bias_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)


def generate_mask_secret_key(
    image_shape: Tuple[int, int, int, int],
    seed: int,
    device: str = 'cpu',
    flip_key_type: str = 'none',
) -> nn.Module:
    """Generates a frozen CNN-based mask generator with cryptographic initialization.
    
    Args:
        image_shape: Input image shape tuple (batch, channels, height, width)
        seed: Random seed for key generation
        device: Target device for the mask generator
    
    Returns:
        Initialized and frozen CNN mask generator
    """
    _, channels, height, width = image_shape
    
    # Generate 256-bit key from seed (for experiment reproduction use)
    random.seed(seed)
    binary_key = random.getrandbits(256).to_bytes(32, 'big')

    # Flip the encryption key (for evaluation)
    if flip_key_type == "none":
        pass
    else:
        binary_key = flip_key(binary_key=binary_key, flip_key_type=flip_key_type)

    # Create and freeze CNN
    mask_generator = CryptoCNN(channels, channels, binary_key).to(device)
    for param in mask_generator.parameters():
        param.requires_grad = False
        
    return mask_generator


def mask_image_with_key(
    images: torch.Tensor,
    cnn_key: nn.Module
) -> torch.Tensor:
    """Applies cryptographic masking to input images using a CNN key.
    
    Args:
        images: Input tensor of shape (batch, channels, height, width)
        cnn_key: Initialized CryptoCNN instance
    
    Returns:
        Masked image tensor with same shape as input
    """
    # Normalize to [0, 1] range
    images_min = images.amin(dim=(1, 2, 3), keepdim=True)
    images_max = images.amax(dim=(1, 2, 3), keepdim=True)
    normalized = (images - images_min) / (images_max - images_min + 1e-8)
    
    return cnn_key(normalized)