import random
from hashlib import shake_256

import numpy as np
import torch
import torch.nn as nn


def generate_keys(length: int, seed: int) -> torch.Tensor:
    """Generates Bernoulli-distributed authentication keys with a given seed.
    
    Args:
        length: Length of the key tensor to generate
        seed: Seed value for reproducible random number generation
    
    Returns:
        Tensor containing binary authentication keys (0s and 1s)
    """
    torch.manual_seed(seed)
    return torch.bernoulli(torch.full((length,), 0.5))


class CryptoCNN(nn.Module):
    """CNN with cryptographic parameter initialization for image masking.
    
    Attributes:
        conv1-5: Convolutional layers with ReLU activations
        relu: ReLU activation function
    """

    def __init__(self, input_channels: int, output_channels: int, binary_key: bytes):
        """Initializes CNN parameters using cryptographic hash of binary key.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            binary_key: 256-bit secret key as bytes for parameter generation
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Initialize parameters using cryptographic key
        layers = [
            ('conv1', self.conv1),
            ('conv2', self.conv2),
            ('conv3', self.conv3),
            ('conv4', self.conv4),
            ('conv5', self.conv5),
        ]
        for name, layer in layers:
            self._init_layer(layer, name, binary_key)

    def _init_layer(self, layer: nn.Module, layer_name: str, binary_key: bytes) -> None:
        """Initializes layer parameters using cryptographic hash function.
        
        Args:
            layer: Layer module to initialize
            layer_name: Identifier for layer in hash function
            binary_key: Secret key bytes for hash initialization
        """
        fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        bound = 1.0 / np.sqrt(fan_in)

        # Initialize weights
        weight = self._generate_parameter(
            binary_key, f"{layer_name}.weight", layer.weight.shape, bound
        )
        layer.weight.data = weight.to(layer.weight.device)

        # Initialize biases
        if layer.bias is not None:
            bias = self._generate_parameter(
                binary_key, f"{layer_name}.bias", layer.bias.shape, bound
            )
            layer.bias.data = bias.to(layer.bias.device)

    def _generate_parameter(
        self, 
        binary_key: bytes, 
        identifier: str, 
        shape: tuple, 
        bound: float
    ) -> torch.Tensor:
        """Generates random parameters using SHAKE-256 extendable-output function.
        
        Args:
            binary_key: Secret key bytes for hash initialization
            identifier: Layer parameter identifier
            shape: Tensor shape for generated parameters
            bound: Scaling factor for parameter values
            
        Returns:
            Tensor with cryptographically-generated values in [-bound, bound)
        """
        num_elements = np.prod(shape)
        num_bytes = num_elements * 4  # 4 bytes per float32
        
        # Generate deterministic bytes using key and parameter identifier
        hasher = shake_256()
        hasher.update(identifier.encode() + binary_key)
        bytes_data = hasher.digest(num_bytes)

        # Convert bytes to normalized float32 tensor
        uint32_vals = np.frombuffer(bytes_data, dtype='>u4')
        floats = uint32_vals.astype(np.float32) / np.float32(0xFFFFFFFF)
        scaled_floats = (floats * 2 * bound) - bound
        
        return torch.from_numpy(scaled_floats.reshape(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers with ReLU activations.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output tensor with same spatial dimensions as input
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.conv5(x)


def generate_mask_secret_key(
    image_shape: tuple, 
    seed: int, 
    device: str = 'cpu'
) -> nn.Module:
    """Generates frozen CNN mask generator initialized with cryptographic key.
    
    Args:
        image_shape: Shape of input images (batch, channels, height, width)
        seed: Seed value for reproducible key generation
        device: Target device for CNN parameters
        
    Returns:
        Frozen CNN module for image masking
    """
    _, channels, height, width = image_shape
    
    # Generate 256-bit key from seed
    random.seed(seed)
    binary_key = random.getrandbits(256).to_bytes(32, 'big')

    # Create and freeze CNN
    cnn = CryptoCNN(channels, channels, binary_key).to(device)
    for param in cnn.parameters():
        param.requires_grad = False
        
    return cnn


def mask_image_with_key(
    images: torch.Tensor, 
    cnn_key: nn.Module
) -> torch.Tensor:
    """Applies CNN-based mask to input images after normalization.
    
    Args:
        images: Input tensor of shape (batch, channels, height, width)
        cnn_key: Mask generator CNN module
        
    Returns:
        Masked images with same shape as input
    """
    # Normalize to [0, 1] range per image
    images_min = images.amin(dim=(1, 2, 3), keepdim=True)
    images_max = images.amax(dim=(1, 2, 3), keepdim=True)
    normalized = (images - images_min) / (images_max - images_min + 1e-8)

    return cnn_key(normalized)