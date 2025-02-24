import torch.nn as nn
import lpips

def get_lpips_loss(device):
    return lpips.LPIPS(net="vgg").to(device)

def get_mse_loss():
    return nn.MSELoss()
