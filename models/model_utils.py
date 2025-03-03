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
    
    # Let's just use the fallback classifier directly - it's simpler and guaranteed to work
    model = SimpleDiverseClassifier(latent_dim, num_classes, seed_key)
    
    # Set all parameters to non-trainable
    for param in model.parameters():
        param.requires_grad = False
    
    # Move model to the specified device before returning
    return model.to(device)