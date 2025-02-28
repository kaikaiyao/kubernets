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

    # Set random seed based on rank for different data generation across GPUs
    torch.manual_seed(2024 + rank)

    # Move models to device and set to appropriate modes
    surrogate_decoder.train()
    surrogate_decoder.to(device)
    gan_model.eval()
    gan_model.to(device)
    watermarked_model.eval()
    watermarked_model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adagrad(surrogate_decoder.parameters(), lr=0.0001)

    # Check if GAN model is StyleGAN2
    is_stylegan2_model = is_stylegan2(gan_model)

    # Calculate number of batches (ceiling division) per GPU
    num_batches = (train_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_norm_diff = 0.0
        num_samples = 0

        for batch_idx in range(num_batches):
            # Generate batch of original and watermarked images
            with torch.no_grad():
                # Generate latent vectors
                z = torch.randn(batch_size, latent_dim, device=device)
                if not is_stylegan2_model:
                    z = z.view(batch_size, latent_dim, 1, 1)

                # Generate images
                if is_stylegan2_model:
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                    x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    x_M = gan_model(z)
                    x_M_hat = watermarked_model(z)

                # Constrain watermarked images
                x_M_hat = constrain_image(x_M_hat, x_M, max_delta)

            # Forward pass through surrogate decoder
            k_M = surrogate_decoder(x_M)
            k_M_hat = surrogate_decoder(x_M_hat)

            # Calculate norms
            d_k_M = torch.norm(k_M, dim=1)
            d_k_M_hat = torch.norm(k_M_hat, dim=1)

            # Loss exactly matching the real decoder's training objective
            norm_diff = d_k_M - d_k_M_hat
            loss = ((norm_diff).max() + 1) ** 2

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            epoch_loss += loss.item() * batch_size
            epoch_norm_diff += norm_diff.mean().item() * batch_size
            num_samples += batch_size

            if rank == 0 and batch_idx % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Norm diff: {norm_diff.mean().item():.4f}, "
                    f"d_k_M range: [{d_k_M.min().item():.4f}, {d_k_M.max().item():.4f}], "
                    f"d_k_M_hat range: [{d_k_M_hat.min().item():.4f}, {d_k_M_hat.max().item():.4f}]"
                )

            # Free up GPU memory
            del z, x_M, x_M_hat, k_M, k_M_hat, d_k_M, d_k_M_hat, norm_diff, loss
            torch.cuda.empty_cache()
            gc.collect()

        # Compute global epoch statistics if using DDP
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
                f"Average Loss: {global_avg_loss:.4f}, "
                f"Average Norm Difference: {global_avg_norm_diff:.4f}"
            )

        torch.cuda.empty_cache()
        gc.collect()

    # Save the trained model only from rank 0
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
) -> torch.Tensor:
    """
    Generate random images for the attack.

    Args:
        gan_model (nn.Module): The GAN model to generate images.
        image_attack_size (int): Number of attack images to generate.
        latent_dim (int): Dimension of the latent space.
        device (torch.device): Device for computations.
        batch_size (int, optional): Batch size for generation. Defaults to 100.

    Returns:
        torch.Tensor: Generated attack images.
    """
    image_attack_batches = []
    num_batches = math.ceil(image_attack_size / batch_size)

    gan_model.to(device)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            logging.info(f"Generating attack images, batch index = {batch_idx}")
            current_batch_size = min(batch_size, image_attack_size - batch_idx * batch_size)
            if is_stylegan2(gan_model):
                z = torch.randn((current_batch_size, latent_dim), device=device)
                x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                x_M = gan_model(z)

            # image_attack_batches.append(torch.rand_like(x_M).cpu()) # 1. use random images
            image_attack_batches.append(x_M.cpu())  # 2. use GAN images directly

            del z, x_M
            torch.cuda.empty_cache()
            gc.collect()

    image_attack = torch.cat(image_attack_batches).to(device)
    del image_attack_batches
    torch.cuda.empty_cache()
    gc.collect()

    return image_attack


def perform_pgd_attack(
    attack_type: str,
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    image_attack: torch.Tensor,
    max_delta: float,
    device: torch.device,
    num_steps: int,
    alpha_values: list,
    momentum: float = 0.8,
    attack_batch_size: int = 10,
    num_transforms: int = 0,
) -> tuple: # enabled momentum and transforms
    surrogate_decoder.eval()
    decoder.eval()
    for param in surrogate_decoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    criterion = nn.BCELoss()

    k_attack_scores_mean = []
    k_attack_scores_std = []

    num_attack_batches = (image_attack.size(0) + attack_batch_size - 1) // attack_batch_size

    for alpha in alpha_values:
        logging.info(f"Performing DI-MIM attack with alpha = {alpha}, momentum = {momentum}")
        k_attack_scores_alpha = []

        for batch_idx in range(num_attack_batches):
            start_idx = batch_idx * attack_batch_size
            end_idx = min((batch_idx + 1) * attack_batch_size, image_attack.size(0))
            image_attack_batch = image_attack[start_idx:end_idx].clone().detach().to(device)
            image_attack_batch.requires_grad = True
            original_images = image_attack_batch.clone().detach()
            target_labels = torch.ones(image_attack_batch.size(0), 1, device=device)

            velocity = torch.zeros_like(image_attack_batch)

            for step in range(num_steps):
                if image_attack_batch.grad is not None:
                    image_attack_batch.grad.zero_()

                # Compute gradients over multiple transformations
                total_grad = torch.zeros_like(image_attack_batch)
                for _ in range(num_transforms):
                    # Random resize (e.g., 90-110% of original size)
                    scale = torch.rand(1).item() * 0.2 + 0.9  # 0.9 to 1.1
                    new_size = int(256 * scale)
                    transformed = F.interpolate(image_attack_batch, size=(new_size, new_size), mode='bilinear', align_corners=False)
                    transformed = F.interpolate(transformed, size=(256, 256), mode='bilinear', align_corners=False)
                    transformed.requires_grad = True
                    
                    outputs = surrogate_decoder(transformed)
                    loss = criterion(outputs, target_labels)
                    loss.backward()
                    total_grad += transformed.grad
                    
                with torch.no_grad():
                    avg_grad = total_grad / num_transforms
                    velocity = momentum * velocity + avg_grad
                    image_attack_batch = image_attack_batch + alpha * velocity.sign()
                    image_attack_batch = torch.clamp(
                        image_attack_batch,
                        min=original_images - max_delta,
                        max=original_images + max_delta,
                    )
                    image_attack_batch.requires_grad = True

                torch.cuda.empty_cache()

            with torch.no_grad():
                if attack_type == "base_baseline":
                    k_attack_batch = decoder(image_attack_batch)
                else:
                    k_mask = generate_mask_secret_key(image_attack_batch.shape, 2024, device=device)
                    k_attack_batch = decoder(mask_image_with_key(image_attack_batch, k_mask))

            k_attack_score_batch = torch.norm(k_attack_batch, dim=1).cpu().numpy()
            k_attack_scores_alpha.extend(k_attack_score_batch)
            logging.info(f"Batch {batch_idx + 1}/{num_attack_batches} processed.")

            del image_attack_batch, target_labels, outputs, loss, k_attack_batch
            torch.cuda.empty_cache()

        mean_score = np.mean(k_attack_scores_alpha)
        std_score = np.std(k_attack_scores_alpha)
        k_attack_scores_mean.append(mean_score)
        k_attack_scores_std.append(std_score)
        logging.info(f"Alpha = {alpha}: k_attack_score mean = {mean_score:.3f}, std = {std_score:.3f}")

    return k_attack_scores_mean, k_attack_scores_std

def attack_label_based(
    attack_type: str,
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    decoder: nn.Module,
    surrogate_decoder: nn.Module,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    image_attack_size: int,
    batch_size: int = 16,
    epochs: int = 1,
    attack_batch_size: int = 16,
    num_steps: int = 500,
    alpha_values: list = None,
    train_surrogate: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """
    Performs a label-based attack on a watermarked GAN model.

    Args:
        attack_type (str): Type of attack to perform.
        gan_model (nn.Module): Original GAN model.
        watermarked_model (nn.Module): Watermarked GAN model.
        max_delta (float): Maximum perturbation.
        decoder (nn.Module): Decoder for scoring.
        surrogate_decoder (nn.Module): Surrogate decoder to train or use.
        latent_dim (int): Latent space dimension.
        device (torch.device): Device for computations.
        train_size (int): Training samples per class.
        image_attack_size (int): Number of attack images.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        epochs (int, optional): Training epochs. Defaults to 1.
        attack_batch_size (int, optional): Batch size for attack. Defaults to 16.
        num_steps (int, optional): PGD steps. Defaults to 100.
        alpha_values (list, optional): Step sizes for PGD. Defaults to None.
        train_surrogate (bool, optional): Whether to train the surrogate decoder. Defaults to True.

    Returns:
        tuple: Mean and standard deviation of attack scores.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if alpha_values is None:
        alpha_values = [0.001]

    logging.info("Starting attack_label_based function.")

    gan_model.eval()
    watermarked_model.eval()
    for param in gan_model.parameters():
        param.requires_grad = False
    for param in watermarked_model.parameters():
        param.requires_grad = False

    # Generate attack images
    image_attack = generate_attack_images(
        gan_model, image_attack_size, latent_dim, device, batch_size=100
    )
    logging.info("Initialized image_attack.")

    # Train surrogate decoder only if train_surrogate is True
    if train_surrogate:
        train_surrogate_decoder(
            attack_type=attack_type,
            surrogate_decoder=surrogate_decoder,
            gan_model=gan_model,
            watermarked_model=watermarked_model,
            max_delta=max_delta,
            latent_dim=latent_dim,
            device=device,
            train_size=train_size,
            epochs=epochs,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            logging.info("Training of surrogate decoder completed.")
    else:
        if rank == 0:
            logging.info("Using pre-trained surrogate decoder.")

    # Perform PGD attack
    if rank == 0:
        k_attack_scores_mean, k_attack_scores_std = perform_pgd_attack(
            attack_type,
            surrogate_decoder, decoder, image_attack, 
            max_delta, device, num_steps, alpha_values, attack_batch_size
        )
        logging.info("PGD attack for different alpha values completed.")

        return k_attack_scores_mean, k_attack_scores_std