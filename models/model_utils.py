import copy
import os
import pickle
import hashlib
import hmac
import numpy as np
import torch
import torch.nn as nn


def save_finetuned_model(model, path, filename):
    model_cpu = copy.deepcopy(model).cpu()
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(model_cpu, f)


def load_finetuned_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def clone_model(model):
    """
    Clones a model and ensures all parameters in the cloned model require gradients.
    
    Args:
        model (torch.nn.Module): The model to clone.
    
    Returns:
        torch.nn.Module: The cloned model with all parameters requiring gradients.
    """
    # Use deepcopy to create a copy of the model
    cloned_model = copy.deepcopy(model)
    
    # Ensure the cloned model is in training mode
    cloned_model.train()
    
    # Ensure all parameters in the cloned model require gradients
    for param in cloned_model.parameters():
        param.requires_grad = True
    
    return cloned_model


def create_z_classifier_model(latent_dim, num_classes, seed_key, device, key_type="csprng"):
    """
    Create a fixed CNN model for classifying latent vectors z into one of N classes.
    
    Args:
        latent_dim: Dimension of the latent vectors
        num_classes: Number of output classes
        seed_key: Seed for CSPRNG
        device: Device to place the model on
        key_type: Type of key generation ("encryption" or "csprng")
    
    Returns:
        A fixed (untrainable) CNN model for z classification
    """
    # Convert seed_key to bytes
    key_bytes = hashlib.sha256(str(seed_key).encode()).digest()
    
    # Create a simple MLP for z classification
    model = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    
    # Initialize using CSPRNG
    total_params = sum(p.numel() for p in model.parameters())
    if key_type == "csprng":
        # Generate pseudorandom bytes using HMAC
        def generate_csprng_bytes(key, num_bytes):
            result = bytearray()
            counter = 0
            while len(result) < num_bytes:
                # Use counter as message to HMAC
                counter_bytes = counter.to_bytes(16, byteorder='big')
                h = hmac.new(key, counter_bytes, hashlib.sha256)
                result.extend(h.digest())
                counter += 1
            return bytes(result[:num_bytes])
        
        # Use CSPRNG to initialize weights
        rng_bytes = generate_csprng_bytes(key_bytes, total_params * 4)  # 4 bytes per float32
        rng_floats = np.frombuffer(rng_bytes, dtype=np.float32)
        
        # Apply fan-in scaling to weights
        param_idx = 0
        for layer in model:
            if isinstance(layer, nn.Linear):
                # Calculate fan_in for Xavier-like scaling
                fan_in = layer.in_features
                weight_scale = np.sqrt(1.0 / fan_in)
                
                # Apply to weights
                num_weights = layer.weight.numel()
                layer_weights = torch.from_numpy(
                    rng_floats[param_idx:param_idx+num_weights].reshape(layer.weight.shape)
                ) * weight_scale
                layer.weight.data.copy_(layer_weights)
                param_idx += num_weights
                
                # Apply to bias
                num_biases = layer.bias.numel()
                layer_biases = torch.from_numpy(
                    rng_floats[param_idx:param_idx+num_biases].reshape(layer.bias.shape)
                ) * 0.1  # smaller scale for biases
                layer.bias.data.copy_(layer_biases)
                param_idx += num_biases
    
    # Set all parameters to non-trainable
    for param in model.parameters():
        param.requires_grad = False
    
    return model.to(device)