import torch
import numpy as np
import math
import logging
import statistics
import matplotlib.pyplot as plt
import lpips
from typing import Tuple

from utils.image_utils import constrain_image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms.functional import to_pil_image
from models.stylegan2 import is_stylegan2
from torchmetrics.image.fid import FrechetInceptionDistance
from key.key import generate_mask_secret_key, mask_image_with_key


def evaluate_model(
    num_images: int,
    gan_model: torch.nn.Module,
    watermarked_model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    plotting: bool,
    latent_dim: int,
    max_delta: float,
    mask_switch_on: bool,
    seed_key: int,
    flip_key_type: str,
    batch_size: int = 8,
    key_type: str = "csprng"
) -> Tuple[float, float, float, float, float, int]:
    """
    Evaluate watermarking model performance with comprehensive metrics
    
    Returns:
        Tuple containing:
        - AUC score
        - TPR@1% FPR
        - Mean LPIPS loss
        - FID score
        - Mean max delta
        - Total decoder parameters
    """
    # Initialize models and metrics
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    
    total_decoder_params = sum(p.numel() for p in decoder.parameters())
    lpips_loss = lpips.LPIPS(net="vgg").to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    # Data collection structures
    metrics = {
        'scores': [],
        'labels': [],
        'original_scores': [],
        'watermarked_scores': [],
        'random_scores': [],
        'lpips_losses': [],
        'max_deltas': [],
        'plot_data': []
    }

    # Process images in batches
    num_batches = math.ceil(num_images / batch_size)
    for batch_idx in range(num_batches):
        process_batch(
            batch_idx=batch_idx,
            num_batches=num_batches,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            decoder=decoder,
            device=device,
            latent_dim=latent_dim,
            max_delta=max_delta,
            mask_switch_on=mask_switch_on,
            seed_key=seed_key,
            batch_size=batch_size,
            lpips_loss=lpips_loss,
            fid_metric=fid_metric,
            metrics=metrics,
            plotting=plotting,
            flip_key_type=flip_key_type,
            num_images=num_images,
            key_type=key_type
        )

    # Calculate final metrics
    results = calculate_final_metrics(metrics, fid_metric)
    generate_plots(metrics, results['auc'], plotting)
    
    return (
        results['auc'],
        results['tpr_at_1_fpr'],
        results['mean_lpips'],
        results['fid_score'],
        results['mean_max_delta'],
        total_decoder_params,
    )

def process_batch(
    batch_idx: int,
    num_batches: int,
    gan_model: torch.nn.Module,
    watermarked_model: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    latent_dim: int,
    max_delta: float,
    mask_switch_on: bool,
    seed_key: int,
    batch_size: int,
    lpips_loss: torch.nn.Module,
    fid_metric: FrechetInceptionDistance,
    metrics: dict,
    plotting: bool,
    flip_key_type: str,
    num_images: int,
    key_type: str = "csprng"
) -> None:
    """Process a single batch of images"""
    current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
    logging.info(f"Processing batch {batch_idx+1}/{num_batches} ({current_batch_size} images)")

    # Generate images
    with torch.no_grad():
        z, x_M, x_M_hat = generate_images(
            gan_model, watermarked_model, current_batch_size, latent_dim, device
        )
        x_M_hat = constrain_image(x_M_hat, x_M, max_delta)
        x_rand = torch.rand_like(x_M) * 2 - 1  # Random images in [-1, 1] range

    # Calculate image differences
    delta_metrics = calculate_delta_metrics(x_M, x_M_hat, current_batch_size, lpips_loss)
    metrics['max_deltas'].extend(delta_metrics['max_deltas'])
    metrics['lpips_losses'].extend(delta_metrics['lpips_losses'])

    # Process watermark detection
    process_watermark_detection(
        x_M, x_M_hat, x_rand, decoder, mask_switch_on, seed_key, device, metrics, current_batch_size, flip_key_type, key_type
    )

    # Update FID metric
    update_fid_metric(x_M, x_M_hat, fid_metric)

    # Store plot data if needed
    if plotting:
        store_plot_data(x_M, x_M_hat, metrics, current_batch_size)

def generate_images(gan_model, watermarked_model, batch_size, latent_dim, device):
    """Generate original and watermarked images"""
    if is_stylegan2(gan_model):
        z = torch.randn((batch_size, latent_dim), device=device)
        x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
        x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
    else:
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        x_M = gan_model(z)
        x_M_hat = watermarked_model(z)
    return z, x_M, x_M_hat

def calculate_delta_metrics(x_M, x_M_hat, batch_size, lpips_loss):
    """Calculate image difference metrics"""
    delta = x_M_hat - x_M
    abs_delta = torch.abs(delta)
    
    return {
        'max_deltas': abs_delta.view(batch_size, -1).max(dim=1)[0].tolist(),
        'lpips_losses': lpips_loss(x_M_hat, x_M).squeeze().tolist()
    }

def process_watermark_detection(x_M, x_M_hat, x_rand, decoder, mask_switch_on, seed_key, device, metrics, batch_size, flip_key_type, key_type="csprng"):
    """Process watermark detection and score calculation for original, watermarked, and random images"""
    # Apply mask if enabled
    if mask_switch_on:
        k_mask = generate_mask_secret_key(x_M_hat.shape, seed_key, device=device, flip_key_type=flip_key_type, key_type=key_type)
        x_M_mask = mask_image_with_key(x_M, k_mask)
        x_M_hat_mask = mask_image_with_key(x_M_hat, k_mask)
        x_rand_mask = mask_image_with_key(x_rand, k_mask)
    else:
        x_M_mask, x_M_hat_mask = x_M, x_M_hat
        x_rand_mask = x_rand

    # Decode watermarks
    with torch.no_grad():
        k_M = decoder(x_M_mask)
        k_M_hat = decoder(x_M_hat_mask)
        k_rand = decoder(x_rand_mask)

    # Calculate similarity scores
    k_M_scores = torch.norm(k_M, dim=1)
    k_M_hat_scores = torch.norm(k_M_hat, dim=1)
    k_rand_scores = torch.norm(k_rand, dim=1)

    # Store scores and labels
    for j in range(batch_size):
        metrics['scores'].extend([k_M_scores[j].item(), k_M_hat_scores[j].item(), k_rand_scores[j].item()])
        metrics['labels'].extend([0, 1, 0])
        metrics['original_scores'].append(k_M_scores[j].item())
        metrics['watermarked_scores'].append(k_M_hat_scores[j].item())
        metrics['random_scores'].append(k_rand_scores[j].item())

def update_fid_metric(x_M, x_M_hat, fid_metric):
    """Update FID metric with new images"""
    x_M_normalized = ((x_M + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    x_M_hat_normalized = ((x_M_hat + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    fid_metric.update(x_M_normalized, real=True)
    fid_metric.update(x_M_hat_normalized, real=False)

def store_plot_data(x_M, x_M_hat, metrics, batch_size):
    """Store image data for later plotting"""
    for j in range(batch_size):
        img_orig = ((x_M[j] + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        img_water = ((x_M_hat[j] + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        difference_image = ((x_M_hat[j] - x_M[j] + 2) / 4 * 255).clamp(0, 255).to(torch.uint8)
        
        metrics['plot_data'].append({
            "orig": img_orig.cpu(),
            "water": img_water.cpu(),
            "difference": difference_image.cpu(),
            "scores": (
                metrics['original_scores'][-batch_size+j],
                metrics['watermarked_scores'][-batch_size+j]
            )
        })

def calculate_final_metrics(metrics, fid_metric) -> dict:
    """Calculate all final metrics from collected data"""
    logging.info("Computing final metrics...")
    
    # Basic statistics
    score_stats = {
        'original_mean': np.mean(metrics['original_scores']),
        'original_std': np.std(metrics['original_scores']),
        'watermarked_mean': np.mean(metrics['watermarked_scores']),
        'watermarked_std': np.std(metrics['watermarked_scores']),
        'random_mean': np.mean(metrics['random_scores']),
        'random_std': np.std(metrics['random_scores'])
    }
    
    logging.info(
        "Confidence score statistics:\n"
        f"  Original:    μ={score_stats['original_mean']:.4f} ±{score_stats['original_std']:.4f}\n"
        f"  Watermarked: μ={score_stats['watermarked_mean']:.4f} ±{score_stats['watermarked_std']:.4f}\n"
        f"  Random:      μ={score_stats['random_mean']:.4f} ±{score_stats['random_std']:.4f}"
    )

    # ROC metrics
    auc = roc_auc_score(metrics['labels'], metrics['scores'])
    fpr, tpr, thresholds = roc_curve(metrics['labels'], metrics['scores'])
    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)

    return {
        'auc': auc,
        'tpr_at_1_fpr': tpr_at_1_fpr,
        'mean_lpips': statistics.mean(metrics['lpips_losses']),
        'mean_max_delta': statistics.mean(metrics['max_deltas']),
        'fid_score': fid_metric.compute(),
        **score_stats
    }

def generate_plots(metrics: dict, auc: float, plotting: bool) -> None:
    """Generate all evaluation plots"""
    if not plotting or not metrics['plot_data']:
        return

    logging.info("Generating evaluation plots...")
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['scores'])
    plt.plot(fpr, tpr, label=f"Watermark Detector (AUC = {auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # Image comparison plot
    plot_images = min(5, len(metrics['plot_data']))
    fig, axs = plt.subplots(plot_images, 3, figsize=(15, plot_images*5))
    for i in range(plot_images):
        data = metrics['plot_data'][i]
        axs[i,0].imshow(to_pil_image(data['orig']))
        axs[i,0].set_title(f"Original\nScore: {data['scores'][0]:.3f}")
        axs[i,1].imshow(to_pil_image(data['water']))
        axs[i,1].set_title(f"Watermarked\nScore: {data['scores'][1]:.3f}")
        axs[i,2].imshow(to_pil_image(data['difference']))
        axs[i,2].set_title("Difference")
    plt.savefig("image_comparison.png")
    plt.close()