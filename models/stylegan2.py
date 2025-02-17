import os
import sys

sys.path.append("./stylegan2-ada-pytorch")
import dnnlib
import legacy

from utils.file_utils import download_file


def load_stylegan2_model(url, local_path, device="cuda"):
    if not os.path.exists(local_path):
        print("Model not found locally. Downloading...")
        download_file(url, local_path)
    with dnnlib.util.open_url(local_path) as f:
        model = legacy.load_network_pkl(f)["G_ema"].to(device)
    return model


def is_stylegan2(model):
    return type(model).__module__ == "torch_utils.persistence" and type(model).__name__ == "Generator"