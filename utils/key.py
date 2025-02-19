import random
import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import ChaCha20  # Requires pycryptodome

class CryptoCNN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, binary_key: bytes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Initialize all layers with ChaCha20 encryption + fan-in scaling
        self._init_layers_with_scaling(binary_key)

    def _init_layers_with_scaling(self, binary_key: bytes) -> None:
        # Calculate total parameters in the network
        total_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        seed_size = total_params * 4  # 4 bytes per float32
        rng = np.random.default_rng(seed=0)
        seed = rng.bytes(seed_size)
        
        # Encrypt seed with ChaCha20
        chacha_key = binary_key[:32]  # 256-bit key
        nonce = b'\x00' * 8  # 64-bit nonce (required by ChaCha20)
        cipher = ChaCha20.new(key=chacha_key, nonce=nonce)
        ciphertext = cipher.encrypt(seed)
        
        # Convert to floats in [0, 1)
        uint32_vals = np.frombuffer(ciphertext, dtype='>u4')
        floats = uint32_vals.astype(np.float32) / np.float32(0xFFFFFFFF)
        
        # Split into parameters with per-layer scaling
        ptr = 0
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            # Calculate fan-in and bound for weights
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            bound = 1.0 / np.sqrt(fan_in)
            
            # Assign weights
            weight_size = np.prod(layer.weight.shape)
            layer_weights = floats[ptr:ptr + weight_size]
            scaled_weights = (layer_weights * 2 * bound) - bound  # Scale to [-bound, bound)
            layer.weight.data = torch.from_numpy(scaled_weights.reshape(layer.weight.shape))
            ptr += weight_size
            
            # Assign biases (if exists)
            if layer.bias is not None:
                bias_size = np.prod(layer.bias.shape)
                layer_biases = floats[ptr:ptr + bias_size]
                scaled_biases = (layer_biases * 2 * bound) - bound  # Same scaling as weights
                layer.bias.data = torch.from_numpy(scaled_biases.reshape(layer.bias.shape))
                ptr += bias_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unchanged forward pass
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
    """Generates frozen CNN mask generator."""
    _, channels, height, width = image_shape
    
    # Generate 256-bit key from seed
    random.seed(seed)
    binary_key = random.getrandbits(256).to_bytes(32, 'big')

    # Create and freeze CNN
    cnn = CryptoCNN(channels, channels, binary_key).to(device)
    for param in cnn.parameters():
        param.requires_grad = False
        
    return cnn

def mask_image_with_key(images: torch.Tensor, cnn_key: nn.Module) -> torch.Tensor:
    # Same masking function as before
    images_min = images.amin(dim=(1, 2, 3), keepdim=True)
    images_max = images.amax(dim=(1, 2, 3), keepdim=True)
    normalized = (images - images_min) / (images_max - images_min + 1e-8)
    return cnn_key(normalized)