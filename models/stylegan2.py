import os
import sys

sys.path.append("./stylegan2-ada-pytorch")
import dnnlib
import legacy
import torch

from utils.file_utils import download_file


def load_stylegan2_model(url, local_path, device):
    if not os.path.exists(local_path):
        download_file(url, local_path)
    
    # Force initialization on CPU first
    with torch.device('cpu'):
        with dnnlib.util.open_url(local_path) as f:
            model = legacy.load_network_pkl(f)["G_ema"]
    
    # Move to GPU incrementally
    model = model.to(device)
    return model

def is_stylegan2(model):
    return type(model).__module__ == "torch_utils.persistence" and type(model).__name__ == "Generator"