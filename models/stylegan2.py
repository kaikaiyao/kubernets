import os
import sys
import torch
import dnnlib
import legacy
from torch.distributed import is_initialized, get_rank
from utils.file_utils import download_file

sys.path.append("./stylegan2-ada-pytorch")

def load_stylegan2_model(url, local_path, device):
    # Only download from rank 0 to avoid conflicts
    if not os.path.exists(local_path):
        if is_initialized() and get_rank() == 0 or not is_initialized():
            print(f"[Rank {get_rank()}] Downloading StyleGAN2 model...")
            download_file(url, local_path)
        
        # Wait for rank 0 to finish downloading
        if is_initialized():
            torch.distributed.barrier()

    # Verify file integrity
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model file not found at {local_path}")
    
    file_size = os.path.getsize(local_path)
    if file_size < 1024:  # Basic sanity check
        raise ValueError(f"Downloaded file seems too small ({file_size} bytes)")

    try:
        # Force initialization on CPU first
        with torch.device('cpu'):
            with dnnlib.util.open_url(local_path) as f:
                model = legacy.load_network_pkl(f)["G_ema"]
        
        # Move to GPU incrementally
        model = model.to(device)
        return model
    
    except Exception as e:
        # Clean up corrupted file
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(f"Failed to load StyleGAN2 model: {str(e)}")

def is_stylegan2(model):
    return type(model).__module__ == "torch_utils.persistence" and type(model).__name__ == "Generator"