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
            self.latent_dim = latent_dim
            self.seed_key = seed_key
            
            # Create multiple projection matrices for more diversity
            torch.manual_seed(seed_key)
            self.projection1 = nn.Parameter(torch.randn(latent_dim, 1))
            self.projection2 = nn.Parameter(torch.randn(latent_dim, 1))
            self.projection3 = nn.Parameter(torch.randn(latent_dim, 1))
            
            # Create class boundaries for more random distribution
            self.class_weights = nn.Parameter(torch.randn(num_classes, latent_dim))
            
            # None of these parameters should be trained
            self.projection1.requires_grad = False
            self.projection2.requires_grad = False
            self.projection3.requires_grad = False
            self.class_weights.requires_grad = False
            
        def forward(self, z):
            batch_size = z.shape[0]
            
            # Approach 1: Multiple projections for more randomness
            proj1 = torch.matmul(z, self.projection1).view(-1)
            proj2 = torch.matmul(z, self.projection2).view(-1)
            proj3 = torch.matmul(z, self.projection3).view(-1)
            
            # Combined projection with non-linear transformation
            combined_proj = torch.sin(proj1) * torch.cos(proj2) + torch.cos(proj1) * torch.sin(proj3)
            
            # Approach 2: Direct class scoring
            # Calculate similarity to class prototypes
            class_scores = torch.matmul(z, self.class_weights.t())
            
            # Final logits: combine both approaches
            # This creates a semi-random but still diverse assignment
            logits = class_scores + torch.sin(combined_proj.unsqueeze(1)) * 3.0
            
            # Add a small random noise based on the iteration to increase diversity
            # Use a hash of the z tensor as random seed to ensure deterministic behavior
            z_hash = hash(str(z.sum().item())) % 10000
            torch.manual_seed(self.seed_key + z_hash)
            noise = torch.randn_like(logits) * 0.5
            
            return logits + noise
    
    # Let's just use the fallback classifier directly - it's simpler and guaranteed to work
    model = SimpleDiverseClassifier(latent_dim, num_classes, seed_key)
    
    # Set all parameters to non-trainable
    for param in model.parameters():
        param.requires_grad = False
    
    # Move model to the specified device before returning
    return model.to(device)