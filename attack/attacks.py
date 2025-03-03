import os
import gc
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from models.stylegan2 import is_stylegan2
from utils.image_utils import constrain_image
from utils.file_utils import generate_time_based_string
from key.key import generate_mask_secret_key, mask_image_with_key
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lpips

def train_surrogate_decoder(
    attack_type: str,
    surrogate_decoder: nn.Module,
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    epochs: int = 5,
    batch_size: int = 16,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """
    Train the surrogate decoder using a loss function similar to the real decoder's training objective.
    """
    time_string = generate_time_based_string()
    if rank == 0:
        logging.info(f"time_string = {time_string}")

    torch.manual_seed(2024 + rank)
    surrogate_decoder.train()
    surrogate_decoder.to(device)
    gan_model.eval()
    gan_model.to(device)
    watermarked_model.eval()
    watermarked_model.to(device)

    optimizer = torch.optim.Adagrad(surrogate_decoder.parameters(), lr=0.0001)
    is_stylegan2_model = is_stylegan2(gan_model)
    num_batches = (train_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_norm_diff = 0.0
        num_samples = 0

        for batch_idx in range(num_batches):
            with torch.no_grad():
                z = torch.randn(batch_size, latent_dim, device=device)
                if not is_stylegan2_model:
                    z = z.view(batch_size, latent_dim, 1, 1)
                if is_stylegan2_model:
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                    x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    x_M = gan_model(z)
                    x_M_hat = watermarked_model(z)
                x_M_hat = constrain_image(x_M_hat, x_M, max_delta)

            k_M = surrogate_decoder(x_M)
            k_M_hat = surrogate_decoder(x_M_hat)
            d_k_M = torch.norm(k_M, dim=1)
            d_k_M_hat = torch.norm(k_M_hat, dim=1)
            norm_diff = d_k_M - d_k_M_hat
            loss = ((norm_diff).max() + 1) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            epoch_norm_diff += norm_diff.mean().item() * batch_size
            num_samples += batch_size

            if rank == 0 and batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, Norm diff: {norm_diff.mean().item():.4f}, "
                    f"d_k_M range: [{d_k_M.min().item():.4f}, {d_k_M.max().item():.4f}], "
                    f"d_k_M_hat range: [{d_k_M_hat.min().item():.4f}, {d_k_M_hat.max().item():.4f}]"
                )

            del z, x_M, x_M_hat, k_M, k_M_hat, d_k_M, d_k_M_hat, norm_diff, loss
            torch.cuda.empty_cache()
            gc.collect()

        if world_size > 1:
            local_avg_loss = epoch_loss / num_samples
            local_avg_loss_tensor = torch.tensor(local_avg_loss, device=device)
            dist.all_reduce(local_avg_loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = local_avg_loss_tensor.item() / world_size

            local_avg_norm_diff = epoch_norm_diff / num_samples
            local_avg_norm_diff_tensor = torch.tensor(local_avg_norm_diff, device=device)
            dist.all_reduce(local_avg_norm_diff_tensor, op=dist.ReduceOp.SUM)
            global_avg_norm_diff = local_avg_norm_diff_tensor.item() / world_size
        else:
            global_avg_loss = epoch_loss / num_samples
            global_avg_norm_diff = epoch_norm_diff / num_samples

        if rank == 0:
            logging.info(
                f"Epoch {epoch + 1}/{epochs} Completed. "
                f"Average Loss: {global_avg_loss:.4f}, Average Norm Difference: {global_avg_norm_diff:.4f}"
            )

        torch.cuda.empty_cache()
        gc.collect()

    if rank == 0:
        model_filename = f"surrogate_decoder_{time_string}.pt"
        if world_size > 1:
            torch.save(surrogate_decoder.module.state_dict(), model_filename)
        else:
            torch.save(surrogate_decoder.state_dict(), model_filename)
        logging.info(f"Surrogate Decoder model saved as {model_filename}")

def generate_attack_images(
    gan_model: nn.Module,
    image_attack_size: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 100,
    attack_image_type: str = "original_image",
) -> torch.Tensor:
    """
    Generate images for the attack.
    
    Args:
        gan_model: The GAN model to use for generating images
        image_attack_size: Number of images to generate
        latent_dim: Dimension of the latent space
        device: Device to use for computation
        batch_size: Batch size for generation
        attack_image_type: Type of images to generate ("original_image", "random_image", or "blurred_image")
    
    Returns:
        torch.Tensor: Generated attack images
    """
    image_attack_batches = []
    num_batches = math.ceil(image_attack_size / batch_size)
    gan_model.to(device)

    with torch.no_grad():
        # Get image shape from GAN model by doing a single forward pass
        if is_stylegan2(gan_model):
            z = torch.randn((1, latent_dim), device=device)
            sample_image = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            z = torch.randn(1, latent_dim, 1, 1, device=device)
            sample_image = gan_model(z)
        image_shape = sample_image.shape[1:]
        del z, sample_image

        # Define a custom Gaussian blur function
        def apply_gaussian_blur(images, kernel_size=21, sigma=7.0):
            # Create a Gaussian kernel
            def gaussian_kernel(kernel_size, sigma=1.0):
                x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
                x = x.view(1, -1).expand(kernel_size, -1)
                y = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
                y = y.view(-1, 1).expand(-1, kernel_size)
                coefficient = 1.0 / (2 * math.pi * sigma**2)
                kernel = coefficient * torch.exp(-(x**2 + y**2) / (2 * sigma**2))
                return kernel / torch.sum(kernel)
            
            # Create a 2D Gaussian kernel
            kernel = gaussian_kernel(kernel_size, sigma)
            
            # Expand to 3 channels
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.repeat(3, 1, 1, 1)
            
            # Apply padding to maintain the same image size
            padding = (kernel_size - 1) // 2
            
            # Apply the kernel to each channel separately to maintain correct color
            blurred_images = F.conv2d(
                images, 
                weight=kernel.to(images.device), 
                padding=padding,
                groups=3  # Apply separately to each channel
            )
            
            return blurred_images

        for batch_idx in range(num_batches):
            logging.info(f"Generating attack images, batch index = {batch_idx}")
            current_batch_size = min(batch_size, image_attack_size - batch_idx * batch_size)
            
            if attack_image_type == "original_image" or attack_image_type == "blurred_image":
                # Generate images using the GAN model
                if is_stylegan2(gan_model):
                    z = torch.randn((current_batch_size, latent_dim), device=device)
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                    x_M = gan_model(z)
                
                if attack_image_type == "blurred_image":
                    # Apply strong Gaussian blur to make images look like from a worse model
                    x_M = apply_gaussian_blur(x_M, kernel_size=21, sigma=7.0)
                    logging.info(f"Applied Gaussian blur to batch {batch_idx}")
                
                image_attack_batches.append(x_M.cpu())
                del z, x_M
            else:  # random_image
                # Generate random noise images with same shape as GAN output
                random_images = torch.rand((current_batch_size,) + image_shape, device=device) * 2 - 1  # Scale to [-1, 1]
                image_attack_batches.append(random_images.cpu())
                del random_images
            
            torch.cuda.empty_cache()
            gc.collect()

    image_attack = torch.cat(image_attack_batches).to(device)
    del image_attack_batches
    torch.cuda.empty_cache()
    gc.collect()
    return image_attack

def generate_initial_perturbations(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,  # Reduced steps
    alpha: float = 0.2,  # Increased step size
    max_delta: float = 2.0,
    early_stop_threshold: float = 1e-6
) -> tuple:
    surrogate_decoder.eval()
    decoder.eval()
    images = images.clone().detach().to(device)
    images.requires_grad = True
    original_images = images.clone().detach()
    
    # Get initial k_scores from real decoder
    with torch.no_grad():
        k_orig = decoder(original_images)
        k_scores_orig = torch.norm(k_orig, dim=1)
        initial_direction = F.normalize(k_orig, dim=1)
        initial_norm = k_scores_orig.mean()

    best_perturbation = None
    best_k_scores = None
    min_k_score = float('inf')
    prev_k_score = float('inf')
    plateau_count = 0

    for step in range(num_steps):
        if images.grad is not None:
            images.grad.zero_()
        
        # Get surrogate decoder's output
        k_surrogate = surrogate_decoder(images)
        k_surrogate_norm = torch.norm(k_surrogate, dim=1)
        
        # Compute real decoder output for loss calculation
        with torch.no_grad():
            k_real = decoder(images)
            k_real_norm = torch.norm(k_real, dim=1)
            current_direction = F.normalize(k_real, dim=1)
        
        # Combined loss to maximize norm reduction and direction change
        norm_reduction = F.relu(k_real_norm - k_scores_orig.mean())  # Encourage norm reduction
        direction_change = 1 - F.cosine_similarity(current_direction, initial_direction, dim=1)
        loss = -(norm_reduction.mean() + direction_change.mean())
        
        loss.backward()
        
        with torch.no_grad():
            # Normalize gradients for stable updates
            grad_norm = torch.norm(images.grad.view(images.size(0), -1), dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
            normalized_grad = images.grad / grad_norm
            
            # Update images with normalized gradient
            images = images - alpha * normalized_grad
            
            # Project back to valid perturbation range
            delta = images - original_images
            delta = torch.clamp(delta, -max_delta, max_delta)
            images = original_images + delta
            images = torch.clamp(images, -1, 1)
            images.requires_grad = True
            
            # Evaluate real decoder's output
            k_pert = decoder(images)
            k_scores_pert = torch.norm(k_pert, dim=1)
            
            # Keep track of best perturbation
            current_k_score = k_scores_pert.mean().item()
            if current_k_score < min_k_score:
                min_k_score = current_k_score
                best_perturbation = images.clone().detach()
                best_k_scores = k_scores_pert.clone().detach()
            
            # Early stopping check
            if abs(current_k_score - prev_k_score) < early_stop_threshold:
                plateau_count += 1
                if plateau_count >= 5:
                    break
            else:
                plateau_count = 0
            prev_k_score = current_k_score
            
            if step % 10 == 0:
                logging.info(
                    f"PGD Step {step}, "
                    f"Original k_scores: {k_scores_orig.mean().item():.4f} "
                    f"[{k_scores_orig.min().item():.4f}, {k_scores_orig.max().item():.4f}], "
                    f"Current k_scores: {k_scores_pert.mean().item():.4f} "
                    f"[{k_scores_pert.min().item():.4f}, {k_scores_pert.max().item():.4f}], "
                    f"Direction Change: {direction_change.mean().item():.4f}, "
                    f"Best k_score: {min_k_score:.4f}"
                )

    return best_perturbation, k_scores_orig, best_k_scores

def fine_tune_surrogate(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 16,
    rank: int = 0
) -> nn.Module:
    """Fine-tune surrogate on perturbed images labeled by real decoder to match output patterns."""
    surrogate_decoder.train()
    decoder.eval()
    
    # Stronger regularization and lower learning rate
    optimizer = torch.optim.Adam(surrogate_decoder.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    num_batches = (images.size(0) + batch_size - 1) // batch_size
    best_loss = float('inf')
    best_model_state = None
    
    # Initialize EMA of real decoder norms
    ema_norm = 0.0
    ema_alpha = 0.1

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_samples = 0

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, images.size(0))
            batch = images[start:end].to(device)
            
            # Generate perturbed images with stronger perturbations
            perturbed_batch, k_scores_orig, k_scores_pert = generate_initial_perturbations(
                surrogate_decoder=surrogate_decoder,
                decoder=decoder,
                images=batch,
                device=device,
                num_steps=50,
                alpha=0.2,
                max_delta=2.0
            )
            
            # Get outputs from both decoders
            with torch.no_grad():
                k_real = decoder(perturbed_batch)
                k_real_orig = decoder(batch)
                
                # Update EMA of real decoder norms
                batch_norm = (torch.norm(k_real, dim=1).mean() + torch.norm(k_real_orig, dim=1).mean()) / 2
                ema_norm = ema_norm * (1 - ema_alpha) + batch_norm.item() * ema_alpha
                
                # Normalize real decoder outputs
                k_real_norm = F.normalize(k_real, dim=1)
                k_real_orig_norm = F.normalize(k_real_orig, dim=1)
            
            # Get surrogate outputs
            k_surrogate = surrogate_decoder(perturbed_batch)
            k_surrogate_orig = surrogate_decoder(batch)
            
            # Normalize surrogate outputs
            k_surrogate_norm = F.normalize(k_surrogate, dim=1)
            k_surrogate_orig_norm = F.normalize(k_surrogate_orig, dim=1)
            
            # 1. Direction alignment loss (primary objective)
            direction_loss = (1 - F.cosine_similarity(k_real_norm, k_surrogate_norm, dim=1)).mean() + \
                           (1 - F.cosine_similarity(k_real_orig_norm, k_surrogate_orig_norm, dim=1)).mean()
            
            # 2. Scale matching loss with EMA norm target
            scale_loss = (F.mse_loss(torch.norm(k_surrogate, dim=1), torch.norm(k_real, dim=1)) + \
                         F.mse_loss(torch.norm(k_surrogate_orig, dim=1), torch.norm(k_real_orig, dim=1)))
            
            # 3. Feature matching loss (reduced weight)
            feature_loss = F.mse_loss(k_surrogate / (torch.norm(k_surrogate, dim=1, keepdim=True) + 1e-8),
                                    k_real / (torch.norm(k_real, dim=1, keepdim=True) + 1e-8)) + \
                         F.mse_loss(k_surrogate_orig / (torch.norm(k_surrogate_orig, dim=1, keepdim=True) + 1e-8),
                                   k_real_orig / (torch.norm(k_real_orig, dim=1, keepdim=True) + 1e-8))
            
            # Combined loss with adjusted weights
            loss = direction_loss + 0.1 * scale_loss + 0.01 * feature_loss
            
            # Add L2 regularization to prevent collapse
            l2_reg = sum(torch.norm(p) ** 2 for p in surrogate_decoder.parameters())
            loss = loss + 0.001 * l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(surrogate_decoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            num_samples += batch.size(0)

            if rank == 0 and i % 10 == 0:
                logging.info(
                    f"Fine-tune Epoch {epoch+1}, Batch {i+1}/{num_batches}, "
                    f"Total Loss: {loss.item():.4f}, "
                    f"Direction Loss: {direction_loss.item():.4f}, "
                    f"Scale Loss: {scale_loss.item():.4f}, "
                    f"Feature Loss: {feature_loss.item():.4f}, "
                    f"L2 Reg: {l2_reg.item():.4f}, "
                    f"Real Norm EMA: {ema_norm:.4f}, "
                    f"Surrogate Norm Range: [{torch.norm(k_surrogate, dim=1).min().item():.4f}, "
                    f"{torch.norm(k_surrogate, dim=1).max().item():.4f}], "
                    f"PGD k_scores - Original: {k_scores_orig.mean().item():.4f}, "
                    f"Perturbed: {k_scores_pert.mean().item():.4f}"
                )

            del batch, perturbed_batch, k_real, k_real_orig, k_surrogate, k_surrogate_orig
            torch.cuda.empty_cache()
            gc.collect()

        avg_epoch_loss = epoch_loss / num_samples
        if rank == 0:
            logging.info(
                f"Fine-tune Epoch {epoch + 1}/{epochs} Completed. "
                f"Average Loss: {avg_epoch_loss:.4f}"
            )
        
        scheduler.step(avg_epoch_loss)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = surrogate_decoder.state_dict().copy()

        torch.cuda.empty_cache()
        gc.collect()

    # Restore best model
    if best_model_state is not None:
        surrogate_decoder.load_state_dict(best_model_state)

    return surrogate_decoder

def perform_pgd_attack(
    attack_type: str,
    surrogate_decoders: list,
    decoder: nn.Module,
    image_attack: torch.Tensor,
    max_delta: float,
    device: torch.device,
    num_steps: int,
    alpha_values: list,
    attack_batch_size: int = 10,
    momentum: float = 0.9,
    ensemble_weights: list = None,
    key_type: str = "csprng",
) -> tuple:
    """
    Perform PGD attack with optional momentum using multiple surrogate decoders.
    
    Args:
        surrogate_decoders: List of surrogate decoder models
        ensemble_weights: List of weights for each surrogate decoder (default: equal weights)
    """
    for surrogate_decoder in surrogate_decoders:
        surrogate_decoder.eval()
        for param in surrogate_decoder.parameters():
            param.requires_grad = False
    
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

    # Initialize ensemble weights if not provided
    if ensemble_weights is None:
        ensemble_weights = [1.0 / len(surrogate_decoders)] * len(surrogate_decoders)
    
    criterion = nn.BCELoss()
    k_attack_scores_mean = []
    k_attack_scores_std = []
    num_attack_batches = (image_attack.size(0) + attack_batch_size - 1) // attack_batch_size

    for alpha in alpha_values:
        logging.info(f"Performing ensemble PGD attack with alpha = {alpha}, momentum = {momentum}")
        k_attack_scores_alpha = []

        for batch_idx in range(num_attack_batches):
            start_idx = batch_idx * attack_batch_size
            end_idx = min((batch_idx + 1) * attack_batch_size, image_attack.size(0))
            image_attack_batch = image_attack[start_idx:end_idx].clone().detach().to(device)
            image_attack_batch.requires_grad = True
            original_images = image_attack_batch.clone().detach()
            target_labels = torch.ones(image_attack_batch.size(0), 1, device=device)

            momentum_buffer = torch.zeros_like(image_attack_batch)

            for step in range(num_steps):
                if image_attack_batch.grad is not None:
                    image_attack_batch.grad.zero_()
                
                # Compute weighted ensemble gradient
                ensemble_grad = torch.zeros_like(image_attack_batch)
                for surrogate_decoder, weight in zip(surrogate_decoders, ensemble_weights):
                    outputs = surrogate_decoder(image_attack_batch)
                    loss = criterion(outputs, target_labels)
                    loss.backward(retain_graph=True)
                    ensemble_grad += weight * image_attack_batch.grad.clone()
                    image_attack_batch.grad.zero_()

                with torch.no_grad():
                    momentum_buffer = momentum * momentum_buffer + ensemble_grad / torch.norm(ensemble_grad, p=1)
                    image_attack_batch = image_attack_batch - alpha * momentum_buffer.sign()
                    image_attack_batch = torch.clamp(
                        image_attack_batch,
                        min=original_images - max_delta,
                        max=original_images + max_delta,
                    )
                    image_attack_batch.requires_grad = True

                torch.cuda.empty_cache()
                gc.collect()

            with torch.no_grad():
                if attack_type in ["base_baseline"]:
                    k_attack_batch = decoder(image_attack_batch)
                elif attack_type in ["base_secure", "combined_secure", "fixed_secure"]:
                    k_mask = generate_mask_secret_key(image_attack_batch.shape, 2024, device=device, key_type=key_type)
                    k_attack_batch = decoder(mask_image_with_key(image_attack_batch, k_mask))
                else:
                    logging.error("attack_type is undefined.")
                
                # Log surrogate and real decoder outputs for diagnostics
                ensemble_output = torch.zeros_like(surrogate_decoders[0](image_attack_batch))
                for surrogate_decoder, weight in zip(surrogate_decoders, ensemble_weights):
                    ensemble_output += weight * surrogate_decoder(image_attack_batch)
                logging.info(f"Batch {batch_idx + 1}, Ensemble output mean: {ensemble_output.mean().item():.3f}")
                logging.info(f"Batch {batch_idx + 1}, Real output mean: {k_attack_batch.mean().item():.3f}")

            k_attack_score_batch = torch.norm(k_attack_batch, dim=1).cpu().numpy()
            k_attack_scores_alpha.extend(k_attack_score_batch)
            logging.info(f"Batch {batch_idx + 1}/{num_attack_batches} processed.")

            del image_attack_batch, target_labels, ensemble_output, k_attack_batch, momentum_buffer
            torch.cuda.empty_cache()
            gc.collect()

        mean_score = np.mean(k_attack_scores_alpha)
        std_score = np.std(k_attack_scores_alpha)
        k_attack_scores_mean.append(mean_score)
        k_attack_scores_std.append(std_score)
        logging.info(f"Alpha = {alpha}: k_attack_score mean = {mean_score:.3f}, std = {std_score:.3f}")

    return k_attack_scores_mean, k_attack_scores_std

def attack_label_based(
    gan_model,
    watermarked_model,
    decoder,
    z_classifier=None,
    device="cuda:0",
    time_string=None,
    latent_dim=512,
    max_delta=0.01,
    saving_path="results",
    seed_key=2024,
    lr_attack=0.001,
    attack_steps=1000,
    lambda_lpips=0.1,
    lambda_D=1.0,
    n_early_exit_threshold=10,
    use_surrogate=False,
    surrogate_decoder=None,
    batch_size=8,
    num_images=100,
    rank=0,
    world_size=1,
    z_dependant_training=False,
    num_classes=10,
    train_surrogate=True,
    finetune_surrogate=False,
    attack_batch_size=16,
    momentum=0.9,
    attack_image_type="original_image",
    key_type="csprng",
):
    """
    Perform label-based attack on the watermarked model
    
    Args:
        gan_model: Original GAN model
        watermarked_model: Watermarked GAN model
        decoder: Decoder model
        z_classifier: Classifier for z vectors (used in z-dependent training)
        device: Device to use
        time_string: Time string for results
        latent_dim: Dimension of latent space
        max_delta: Maximum pixel-wise difference allowed
        saving_path: Path to save results
        seed_key: Seed for random number generation
        lr_attack: Learning rate for attack
        attack_steps: Number of attack steps
        lambda_lpips: Weight for LPIPS loss
        lambda_D: Weight for decoder loss
        n_early_exit_threshold: Threshold for early stopping
        use_surrogate: Whether to use surrogate model
        surrogate_decoder: Surrogate decoder model
        batch_size: Batch size for attack
        num_images: Number of images to attack
        rank: Process rank
        world_size: World size for distributed training
        z_dependant_training: Whether to use z-dependent training
        num_classes: Number of classes for z-dependent training
        train_surrogate: Whether to train surrogate model
        finetune_surrogate: Whether to fine-tune surrogate model
        attack_batch_size: Batch size for attack
        momentum: Momentum for optimizer
        attack_image_type: Type of attack image
        key_type: Type of key generation
    """
    if rank == 0:
        logging.info(f"Starting label-based attack with {'z-dependent' if z_dependant_training else 'standard'} mode")
        logging.info(f"Parameters: max_delta={max_delta}, lr_attack={lr_attack}, attack_steps={attack_steps}")
        logging.info(f"Lambda_lpips={lambda_lpips}, lambda_D={lambda_D}")
    
    # Generate attack images
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    
    # Generate latent vectors
    torch.manual_seed(seed_key)
    all_z = torch.randn((num_images, latent_dim), device=device)
    
    # For z-dependent training, get the classes
    if z_dependant_training and z_classifier is not None:
        with torch.no_grad():
            z_class_logits = z_classifier(all_z)
            z_classes = torch.argmax(z_class_logits, dim=1)
    else:
        z_classes = None
    
    # Generate original images
    with torch.no_grad():
        if is_stylegan2(gan_model):
            orig_images = gan_model(all_z, None, truncation_psi=1.0, noise_mode="const")
        else:
            orig_images = gan_model(all_z)
    
    # Generate watermarked images
    with torch.no_grad():
        if is_stylegan2(watermarked_model):
            watermarked_images = watermarked_model(all_z, None, truncation_psi=1.0, noise_mode="const")
        else:
            watermarked_images = watermarked_model(all_z)
        
        # Apply constraints
        watermarked_images = constrain_image(watermarked_images, orig_images, max_delta)
    
    # Initialize attack perturbations
    perturbations = torch.zeros_like(watermarked_images, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([perturbations], lr=lr_attack)
    
    # Track best attack
    best_attack_images = watermarked_images.clone().detach()
    best_decoder_scores = torch.ones(num_images, device=device)
    
    # LPIPS loss
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    early_exit_counter = 0
    
    # Attack loop
    for step in range(attack_steps):
        optimizer.zero_grad()
        
        # Apply perturbation
        attacked_images = watermarked_images + perturbations
        
        # Constrain within max_delta
        attacked_images = constrain_image(attacked_images, watermarked_images, max_delta)
        
        # Forward pass through decoder
        with torch.no_grad():
            decoder_output = decoder(attacked_images)
        
        # Calculate decoder loss based on training mode
        if z_dependant_training:
            # Multi-class output
            decoder_probs = torch.softmax(decoder_output, dim=1)
            
            if z_classes is not None:
                # Get confidence scores for true classes
                decoder_scores = torch.gather(decoder_probs, 1, z_classes.unsqueeze(1)).squeeze(1)
            else:
                # Get max confidence scores
                decoder_scores = torch.max(decoder_probs, dim=1)[0]
            
            # Reverse for attack: minimize confidence
            decoder_loss = -torch.mean(torch.log(1 - decoder_scores + 1e-7))
        else:
            # Binary output
            decoder_scores = decoder_output.squeeze(1)
            decoder_loss = -torch.mean(torch.log(1 - decoder_scores + 1e-7))
        
        # LPIPS similarity loss
        lpips_values = lpips_loss_fn(watermarked_images, attacked_images).squeeze()
        lpips_value = torch.mean(lpips_values)
        
        # Total loss
        total_loss = lambda_D * decoder_loss + lambda_lpips * lpips_value
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Update best attack
        with torch.no_grad():
            # Update best attacks for each image
            improved_indices = decoder_scores < best_decoder_scores
            if improved_indices.any():
                best_attack_images[improved_indices] = attacked_images[improved_indices].detach()
                best_decoder_scores[improved_indices] = decoder_scores[improved_indices].detach()
                early_exit_counter = 0
            else:
                early_exit_counter += 1
        
        # Log progress
        if rank == 0 and step % 10 == 0:
            avg_decoder_score = torch.mean(decoder_scores).item()
            avg_best_score = torch.mean(best_decoder_scores).item()
            logging.info(f"Step {step}/{attack_steps}: "
                         f"decoder_loss={decoder_loss.item():.4f}, "
                         f"lpips_value={lpips_value.item():.4f}, "
                         f"avg_score={avg_decoder_score:.4f}, "
                         f"best_avg={avg_best_score:.4f}")
        
        # Early stopping
        if early_exit_counter >= n_early_exit_threshold:
            if rank == 0:
                logging.info(f"Early stopping at step {step} due to no improvement")
            break
    
    # Evaluate final attack
    with torch.no_grad():
        final_decoder_output = decoder(best_attack_images)
        
        if z_dependant_training:
            # Multi-class output
            final_decoder_probs = torch.softmax(final_decoder_output, dim=1)
            
            if z_classes is not None:
                # Get confidence scores for true classes
                final_scores = torch.gather(final_decoder_probs, 1, z_classes.unsqueeze(1)).squeeze(1)
            else:
                # Get max confidence scores
                final_scores = torch.max(final_decoder_probs, dim=1)[0]
        else:
            # Binary output
            final_scores = final_decoder_output.squeeze(1)
    
    # Calculate final metrics
    attack_success_rate = (final_scores < 0.5).float().mean().item()
    avg_confidence = final_scores.mean().item()
    avg_lpips = lpips_loss_fn(watermarked_images, best_attack_images).mean().item()
    
    # Log results
    if rank == 0:
        mode_str = "z-dependent" if z_dependant_training else "standard"
        logging.info(f"Attack results ({mode_str}): "
                     f"Success Rate={attack_success_rate:.4f}, "
                     f"Avg Confidence={avg_confidence:.4f}, "
                     f"Avg LPIPS={avg_lpips:.4f}")
    
    # Save results
    if rank == 0 and saving_path:
        os.makedirs(saving_path, exist_ok=True)
        result_file = os.path.join(saving_path, f"attack_results_{time_string}.txt")
        with open(result_file, "w") as f:
            f.write(f"Attack type: label-based\n")
            f.write(f"Training mode: {mode_str}\n")
            f.write(f"Success rate: {attack_success_rate:.4f}\n")
            f.write(f"Average confidence: {avg_confidence:.4f}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
            f.write(f"Max delta: {max_delta}\n")
            f.write(f"Attack steps: {attack_steps}\n")
    
    return attack_success_rate, avg_confidence, avg_lpips