import torch.nn as nn
import lpips

def get_lpips_loss(device):
    return lpips.LPIPS(net="vgg").to(device)

def get_mse_loss():
    return nn.MSELoss()

def get_key_loss(d_k_M_hat, d_k_M):
    loss = ((d_k_M_hat - d_k_M).max() + 1) ** 2
    return loss
