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
    gan_model,
    watermarked_model,
    decoder,
    z_classifier=None,
    num_images=100,
    device="cuda:0",
    time_string=None,
    latent_dim=512,
    saving_path="results",
    seed_key=2024,
    evaluate_from_checkpoint=False,
    checkpoint_iter=None,
    rank=0,
    world_size=1,
    z_dependant_training=False,
    num_classes=10,
    batch_size=8,
    plotting=True,
    max_delta=0.01,
    mask_switch_on=False,
    flip_key_type="none",
    key_type="csprng",
):
    """
    Evaluate watermarking model performance with comprehensive metrics
    
    Args:
        gan_model: Original GAN model
        watermarked_model: Watermarked GAN model
        decoder: Decoder model for watermark extraction
        z_classifier: Classifier for z vectors (used in z-dependent training)
        num_images: Number of images to use for evaluation
        device: Device to use
        time_string: Time string for saving results
        latent_dim: Dimension of latent space
        saving_path: Path to save results
        seed_key: Seed for random number generation
        evaluate_from_checkpoint: Whether this is being called during checkpoint evaluation
        checkpoint_iter: Iteration of checkpoint
        rank: Process rank for distributed training
        world_size: Total number of processes
        z_dependant_training: Whether to use z-dependent training mode
        num_classes: Number of classes for z-dependent training
        batch_size: Batch size for evaluation
        plotting: Whether to generate plots
        max_delta: Maximum pixel-wise difference allowed
        mask_switch_on: Whether to use image masking
        flip_key_type: Type of key flipping to use
        key_type: Type of key generation
    
    Returns:
        Tuple containing:
        - AUC score
        - TPR@1% FPR
        - Mean LPIPS loss
        - FID score
        - Mean max delta
        - Total decoder parameters
    """
    if rank == 0:
        logging.info(f"Evaluating with {'z-dependent' if z_dependant_training else 'standard'} mode")
        
    # Initialize models and metrics
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    
    total_decoder_params = sum(p.numel() for p in decoder.parameters())
    
    # Initialize LPIPS for perceptual similarity
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    
    # Dictionary to store metrics
    metrics = {
        'original_scores': [],
        'watermarked_scores': [],
        'random_scores': [],
        'lpips_values': [],
        'pixel_max_deltas': [],
        'plot_original_images': [],
        'plot_watermarked_images': []
    }
    
    num_batches = math.ceil(num_images / batch_size)
    
    # Process in batches
    for batch_idx in range(num_batches):
        process_batch(
            batch_idx,
            num_batches,
            gan_model,
            watermarked_model,
            decoder,
            device,
            latent_dim,
            max_delta,
            mask_switch_on,
            seed_key,
            batch_size,
            lpips_loss,
            fid_metric,
            metrics,
            plotting,
            flip_key_type,
            num_images,
            z_classifier, 
            z_dependant_training,
            num_classes,
            key_type
        )
    
    # Calculate final metrics
    results = calculate_final_metrics(metrics, fid_metric)
    auc, tpr_at_1_fpr, mean_lpips, fid_score, mean_max_delta = (
        results['auc'], results['tpr_at_1_fpr'], results['mean_lpips'], 
        results['fid_score'], results['mean_max_delta']
    )
    
    # Log results
    if rank == 0:
        mode_str = "z-dependent" if z_dependant_training else "standard"
        checkpoint_str = f" (checkpoint {checkpoint_iter})" if evaluate_from_checkpoint else ""
        logging.info(
            f"Evaluation results ({mode_str}){checkpoint_str}: "
            f"AUC: {auc:.4f}, TPR@1%FPR: {tpr_at_1_fpr:.4f}, "
            f"LPIPS: {mean_lpips:.4f}, FID: {fid_score:.4f}, "
            f"Mean Max Delta: {mean_max_delta:.4f}"
        )
    
    # Generate plots if requested
    if plotting and rank == 0:
        generate_plots(metrics, auc, plotting)
    
    return auc, tpr_at_1_fpr, mean_lpips, fid_score, mean_max_delta, total_decoder_params

def process_batch(
    batch_idx,
    num_batches,
    gan_model,
    watermarked_model,
    decoder,
    device,
    latent_dim,
    max_delta,
    mask_switch_on,
    seed_key,
    batch_size,
    lpips_loss,
    fid_metric,
    metrics,
    plotting,
    flip_key_type,
    num_images,
    z_classifier,
    z_dependant_training,
    num_classes,
    key_type="csprng"
):
    """Process a batch of images for evaluation"""
    current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
    if current_batch_size <= 0:
        return
    
    # Sample z for this batch
    torch.manual_seed(seed_key + batch_idx)
    z = torch.randn((current_batch_size, latent_dim), device=device)
    
    # Get z classes if using z-dependent training
    if z_dependant_training and z_classifier is not None:
        with torch.no_grad():
            z_class_logits = z_classifier(z)
            z_classes = torch.argmax(z_class_logits, dim=1)
    
    # Generate images
    x_M, x_M_hat, x_rand = generate_images(gan_model, watermarked_model, current_batch_size, latent_dim, device)
    
    # Apply constraint
    x_M_hat = constrain_image(x_M_hat, x_M, max_delta)
    
    # Calculate image differences
    delta_metrics = calculate_delta_metrics(x_M, x_M_hat, current_batch_size, lpips_loss)
    metrics['pixel_max_deltas'].extend(delta_metrics['max_deltas'])
    metrics['lpips_values'].extend(delta_metrics['lpips_losses'])
    
    # Process watermark detection
    process_watermark_detection(
        x_M, x_M_hat, x_rand, decoder, mask_switch_on, seed_key, device, metrics, 
        current_batch_size, flip_key_type, z_classifier, z_dependant_training, 
        z_classes if z_dependant_training else None, key_type
    )
    
    # Update FID metric
    update_fid_metric(x_M, x_M_hat, fid_metric)
    
    # Store plot data
    if plotting:
        store_plot_data(x_M, x_M_hat, metrics, current_batch_size)
    
    # Log progress
    logging.info(f"Processed batch {batch_idx+1}/{num_batches}")

def generate_images(gan_model, watermarked_model, batch_size, latent_dim, device):
    """Generate original and watermarked images"""
    with torch.no_grad():
        # Generate latent vector
        if is_stylegan2(gan_model):
            z = torch.randn((batch_size, latent_dim), device=device)
            x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
            
            # Generate random image with same latent dim for comparison
            z_rand = torch.randn((batch_size, latent_dim), device=device)
            x_rand = gan_model(z_rand, None, truncation_psi=1.0, noise_mode="const")
        else:
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            x_M = gan_model(z)
            x_M_hat = watermarked_model(z)
            
            # Generate random image with same latent dim for comparison
            z_rand = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            x_rand = gan_model(z_rand)
        
        # Verify that all images are 256x256 as expected
        expected_size = (256, 256)
        
        if x_M.shape[2:] != expected_size:
            logging.error(f"ERROR: Original images have unexpected size: {x_M.shape[2]}x{x_M.shape[3]} (expected {expected_size[0]}x{expected_size[1]})")
            
        if x_M_hat.shape[2:] != expected_size:
            logging.error(f"ERROR: Watermarked images have unexpected size: {x_M_hat.shape[2]}x{x_M_hat.shape[3]} (expected {expected_size[0]}x{expected_size[1]})")
            
        if x_rand.shape[2:] != expected_size:
            logging.error(f"ERROR: Random images have unexpected size: {x_rand.shape[2]}x{x_rand.shape[3]} (expected {expected_size[0]}x{expected_size[1]})")
        
    return x_M, x_M_hat, x_rand

def calculate_delta_metrics(x_M, x_M_hat, batch_size, lpips_loss):
    """Calculate image difference metrics"""
    delta = x_M_hat - x_M
    abs_delta = torch.abs(delta)
    
    return {
        'max_deltas': abs_delta.view(batch_size, -1).max(dim=1)[0].tolist(),
        'lpips_losses': lpips_loss(x_M_hat, x_M).squeeze().tolist()
    }

def process_watermark_detection(
    x_M, x_M_hat, x_rand, decoder, mask_switch_on, seed_key, device, metrics, 
    batch_size, flip_key_type, z_classifier=None, z_dependant_training=False, 
    z_classes=None, key_type="csprng"
):
    """Process watermark detection for a batch"""
    # Apply masking if enabled
    if mask_switch_on:
        # Create mask
        k_mask = generate_mask_secret_key(x_M.shape, seed_key, device=device, flip_key_type=flip_key_type, key_type=key_type)
        
        # Apply mask
        x_M = mask_image_with_key(x_M, k_mask)
        x_M_hat = mask_image_with_key(x_M_hat, k_mask)
        x_rand = mask_image_with_key(x_rand, k_mask)
    
    with torch.no_grad():
        # Get watermark detection scores
        d_M = decoder(x_M)
        d_M_hat = decoder(x_M_hat)
        d_rand = decoder(x_rand)
        
        # Process scores based on training mode
        if z_dependant_training:
            # For z-dependent mode, extract confidence scores for the predicted class
            d_M_probs = torch.softmax(d_M, dim=1)
            d_M_hat_probs = torch.softmax(d_M_hat, dim=1)
            d_rand_probs = torch.softmax(d_rand, dim=1)
            
            # Get highest confidence for each image
            if z_classes is not None:
                # Use the actual class for watermarked images
                orig_scores = torch.gather(d_M_probs, 1, z_classes.unsqueeze(1)).squeeze(1)
                water_scores = torch.gather(d_M_hat_probs, 1, z_classes.unsqueeze(1)).squeeze(1)
                # For random, use the predicted class
                rand_preds = torch.argmax(d_rand_probs, dim=1)
                rand_scores = torch.gather(d_rand_probs, 1, rand_preds.unsqueeze(1)).squeeze(1)
            else:
                # If no z_classes, use max probability for all
                orig_scores = torch.max(d_M_probs, dim=1)[0]
                water_scores = torch.max(d_M_hat_probs, dim=1)[0]
                rand_scores = torch.max(d_rand_probs, dim=1)[0]
        else:
            # Standard binary classification with sigmoid outputs
            orig_scores = d_M.squeeze(1)
            water_scores = d_M_hat.squeeze(1)
            rand_scores = d_rand.squeeze(1)
        
        # Store scores
        metrics['original_scores'].extend(orig_scores.cpu().numpy())
        metrics['watermarked_scores'].extend(water_scores.cpu().numpy())
        metrics['random_scores'].extend(rand_scores.cpu().numpy())

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
        
        metrics['plot_original_images'].append({
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
        'mean_lpips': statistics.mean(metrics['lpips_values']),
        'mean_max_delta': statistics.mean(metrics['pixel_max_deltas']),
        'fid_score': fid_metric.compute(),
        **score_stats
    }

def generate_plots(metrics: dict, auc: float, plotting: bool) -> None:
    """Generate all evaluation plots"""
    if not plotting or not metrics['plot_original_images']:
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
    plot_images = min(5, len(metrics['plot_original_images']))
    fig, axs = plt.subplots(plot_images, 3, figsize=(15, plot_images*5))
    for i in range(plot_images):
        data = metrics['plot_original_images'][i]
        axs[i,0].imshow(to_pil_image(data['orig']))
        axs[i,0].set_title(f"Original\nScore: {data['scores'][0]:.3f}")
        axs[i,1].imshow(to_pil_image(data['water']))
        axs[i,1].set_title(f"Watermarked\nScore: {data['scores'][1]:.3f}")
        axs[i,2].imshow(to_pil_image(data['difference']))
        axs[i,2].set_title("Difference")
    plt.savefig("image_comparison.png")
    plt.close()