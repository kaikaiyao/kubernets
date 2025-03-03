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
    
    # Use a simpler fallback classifier that guarantees class diversity
    class SimpleDiverseClassifier(nn.Module):
        def __init__(self, latent_dim, num_classes, seed_key):
            super().__init__()
            self.num_classes = num_classes
            # Use a fixed projection matrix initialized with the seed
            torch.manual_seed(seed_key)
            self.projection = nn.Parameter(torch.randn(latent_dim, 1))
            self.projection.requires_grad = False
            
        def forward(self, z):
            # Project the latent vectors to 1D values
            projections = torch.matmul(z, self.projection).view(-1)
            
            # Map to classes based on value ranges
            # This ensures an approximately uniform distribution across classes
            batch_size = projections.shape[0]
            logits = torch.zeros(batch_size, self.num_classes, device=projections.device)
            
            # Sort projections to determine rank order
            sorted_indices = torch.argsort(projections)
            
            # Assign classes based on rank
            for i, idx in enumerate(sorted_indices):
                # Determine class based on position in sorted list
                assigned_class = (i * self.num_classes) // batch_size
                # Put a high value for the assigned class
                logits[idx, assigned_class] = 10.0
                
            return logits
    
    # Create model
    try:
        # Try the complex model first
        class ZClassifier(nn.Module):
            def __init__(self, latent_dim, num_classes):
                super().__init__()
                hidden_dim = 256
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # Normalize activations to prevent mode collapse
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, num_classes)
                )
                
                # Add a learnable temperature parameter (fixed after initialization)
                self.temperature = nn.Parameter(torch.ones(1) * 2.0)
                
            def forward(self, x):
                logits = self.model(x)
                # Apply temperature scaling to create more uniform class distributions
                return logits / self.temperature
        
        model = ZClassifier(latent_dim, num_classes)
        
        # Initialize using CSPRNG with specific modifications to ensure class diversity
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
            
            # Apply initialization with careful scaling to ensure class diversity
            param_idx = 0
            for name, param in model.named_parameters():
                # Skip the temperature parameter
                if "temperature" in name:
                    continue
                    
                numel = param.numel()
                
                if 'weight' in name:
                    # For final layer, make sure weights have significant variation
                    if param.shape[0] == num_classes:
                        # Scale final layer weights to have larger magnitude to increase class diversity
                        param_data = torch.from_numpy(
                            rng_floats[param_idx:param_idx+numel].reshape(param.shape)
                        ) * 1.0
                        
                        # Ensure each output neuron has distinct patterns
                        for i in range(num_classes):
                            # Add class-specific bias to weights
                            class_offset = (i - num_classes // 2) * 0.2
                            param_data[i] += class_offset
                    else:
                        # For other layers, use standard Xavier scaling
                        # Check if parameter has at least 2 dimensions before accessing shape[1]
                        fan_in = param.shape[1] if len(param.shape) > 1 else param.shape[0]
                        param_data = torch.from_numpy(
                            rng_floats[param_idx:param_idx+numel].reshape(param.shape)
                        ) * np.sqrt(2.0 / fan_in)
                else:  # bias
                    if param.shape[0] == num_classes:
                        # Create explicit bias for each class to improve separation
                        param_data = torch.zeros_like(param)
                        for i in range(num_classes):
                            # Distribute biases evenly around zero
                            param_data[i] = (i - (num_classes - 1) / 2) * 0.5
                    else:
                        # Regular initialization for other biases
                        param_data = torch.from_numpy(
                            rng_floats[param_idx:param_idx+numel].reshape(param.shape)
                        ) * 0.1
                        
                param.data.copy_(param_data)
                param_idx += numel
    except Exception as e:
        print(f"Warning: Failed to create complex classifier: {str(e)}. Using fallback classifier.")
        # Use the fallback classifier if the complex one fails
        model = SimpleDiverseClassifier(latent_dim, num_classes, seed_key)
    
    # Set all parameters to non-trainable
    for param in model.parameters():
        param.requires_grad = False
    
    # Test the model to ensure it's producing diverse outputs
    with torch.no_grad():
        test_z = torch.randn(100, latent_dim, device=device)
        test_output = model(test_z)
        test_classes = torch.argmax(test_output, dim=1)
        class_counts = torch.bincount(test_classes, minlength=num_classes)
        unique_classes = (class_counts > 0).sum().item()
        
        # If less than half of classes are being used, fall back to the simple classifier
        if unique_classes < num_classes / 2:
            print(f"Warning: Complex classifier only using {unique_classes}/{num_classes} classes. Using fallback classifier.")
            model = SimpleDiverseClassifier(latent_dim, num_classes, seed_key)
    
    return model.to(device)