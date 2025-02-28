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
) -> torch.Tensor:
    """
    Generate random images for the attack.
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
            image_attack_batches.append(x_M.cpu())
            del z, x_M
            torch.cuda.empty_cache()
            gc.collect()

    image_attack = torch.cat(image_attack_batches).to(device)
    del image_attack_batches
    torch.cuda.empty_cache()
    gc.collect()
    return image_attack

def generate_initial_perturbations(
    surrogate_decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    num_steps: int = 40,
    alpha: float = 0.01,
    max_delta: float = 0.05
) -> torch.Tensor:
    """Generate perturbed images using basic PGD for fine-tuning."""
    surrogate_decoder.eval()
    images = images.clone().detach().to(device)
    images.requires_grad = True
    original_images = images.clone().detach()
    criterion = nn.BCELoss()
    target_labels = torch.ones(images.size(0), 1, device=device)

    for _ in range(num_steps):
        if images.grad is not None:
            images.grad.zero_()
        outputs = surrogate_decoder(images)
        loss = criterion(outputs, target_labels)
        loss.backward()
        with torch.no_grad():
            images = images - alpha * images.grad.sign()
            images = torch.clamp(images, original_images - max_delta, original_images + max_delta)
            images.requires_grad = True
    return images.detach()

def fine_tune_surrogate(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    rank: int = 0
) -> nn.Module:
    """Fine-tune surrogate on perturbed images labeled by real decoder to minimize the norm difference, generating images on-the-fly."""
    surrogate_decoder.train()
    optimizer = torch.optim.Adagrad(surrogate_decoder.parameters(), lr=0.001)  # Increased learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    num_batches = (images.size(0) + batch_size - 1) // batch_size
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_samples = 0

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, images.size(0))
            batch = images[start:end]
            
            # Generate perturbed images on-the-fly
            perturbed_batch = generate_initial_perturbations(
                surrogate_decoder=surrogate_decoder,
                images=batch,
                device=device,
                num_steps=50,  # Reduced steps for faster iteration
                alpha=0.05,
                max_delta=2.0
            )
            
            # Get outputs from both decoders
            with torch.no_grad():
                k_real = decoder(perturbed_batch)
                # Apply sigmoid to normalize real decoder outputs
                k_real = torch.sigmoid(k_real)
            
            k_surrogate = surrogate_decoder(perturbed_batch)
            # Apply sigmoid to normalize surrogate outputs
            k_surrogate = torch.sigmoid(k_surrogate)
            
            # Calculate normalized norms
            d_k_real = torch.norm(k_real, dim=1)
            d_k_surrogate = torch.norm(k_surrogate, dim=1)
            
            # Compute relative error and MSE loss
            norm_diff = d_k_real - d_k_surrogate
            relative_error = torch.abs(norm_diff) / (d_k_real + 1e-6)
            mse_loss = F.mse_loss(k_surrogate, k_real)
            
            # Combined loss with both norm difference and direct output matching
            loss = torch.mean(relative_error) + mse_loss

            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(surrogate_decoder.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            num_samples += batch.size(0)

            if rank == 0 and i % 10 == 0:
                logging.info(
                    f"Fine-tune Epoch {epoch+1}, Batch {i+1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, "
                    f"Relative Error: {relative_error.mean().item():.4f}, "
                    f"d_k_real range: [{d_k_real.min().item():.4f}, {d_k_real.max().item():.4f}], "
                    f"d_k_surrogate range: [{d_k_surrogate.min().item():.4f}, {d_k_surrogate.max().item():.4f}]"
                )

        avg_epoch_loss = epoch_loss / num_samples
        if rank == 0:
            logging.info(
                f"Fine-tune Epoch {epoch + 1}/{epochs} Completed. "
                f"Average Loss: {avg_epoch_loss:.4f}"
            )
        
        # Learning rate scheduling
        scheduler.step(avg_epoch_loss)
        
        # Save best model
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
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    image_attack: torch.Tensor,
    max_delta: float,
    device: torch.device,
    num_steps: int,
    alpha_values: list,
    attack_batch_size: int = 10,
    momentum: float = 0.9,
) -> tuple:
    """
    Perform PGD attack with optional momentum.
    """
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
        logging.info(f"Performing PGD attack with alpha = {alpha}, momentum = {momentum}")
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
                outputs = surrogate_decoder(image_attack_batch)
                loss = criterion(outputs, target_labels)
                loss.backward()
                with torch.no_grad():
                    grad = image_attack_batch.grad.detach()
                    momentum_buffer = momentum * momentum_buffer + grad / torch.norm(grad, p=1)
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
                    k_mask = generate_mask_secret_key(image_attack_batch.shape, 2024, device=device)
                    k_attack_batch = decoder(mask_image_with_key(image_attack_batch, k_mask))
                else:
                    logging.error("attack_type is undefined.")
                # Log surrogate and real decoder outputs for diagnostics
                surr_output = surrogate_decoder(image_attack_batch)
                logging.info(f"Batch {batch_idx + 1}, Surrogate output mean: {surr_output.mean().item():.3f}")
                logging.info(f"Batch {batch_idx + 1}, Real output mean: {k_attack_batch.mean().item():.3f}")

            k_attack_score_batch = torch.norm(k_attack_batch, dim=1).cpu().numpy()
            k_attack_scores_alpha.extend(k_attack_score_batch)
            logging.info(f"Batch {batch_idx + 1}/{num_attack_batches} processed.")

            del image_attack_batch, target_labels, outputs, loss, k_attack_batch, momentum_buffer, surr_output
            torch.cuda.empty_cache()
            gc.collect()

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
    finetune_surrogate: bool = False,  # New switch parameter
    rank: int = 0,
    world_size: int = 1,
    momentum: float = 0.9,
) -> tuple:
    """
    Performs a label-based attack on a watermarked GAN model with optional surrogate fine-tuning.

    Args:
        finetune_surrogate (bool, optional): Whether to fine-tune the surrogate on real decoder outputs. Defaults to False.
        ... (other args remain unchanged)
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

    # Train surrogate decoder if specified
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

    # Fine-tune surrogate if specified
    if finetune_surrogate and rank == 0:
        surrogate_decoder = fine_tune_surrogate(
            surrogate_decoder=surrogate_decoder,
            decoder=decoder,
            images=image_attack,
            device=device,
            epochs=20,  # Adjustable
            batch_size=batch_size,
            rank=rank
        )
        logging.info("Surrogate decoder fine-tuned with real decoder outputs.")

    # Perform PGD attack
    if rank == 0:
        k_attack_scores_mean, k_attack_scores_std = perform_pgd_attack(
            attack_type=attack_type,
            surrogate_decoder=surrogate_decoder,
            decoder=decoder,
            image_attack=image_attack,
            max_delta=max_delta,
            device=device,
            num_steps=num_steps,
            alpha_values=alpha_values,
            attack_batch_size=attack_batch_size,
            momentum=momentum
        )
        logging.info("PGD attack for different alpha values completed.")
        return k_attack_scores_mean, k_attack_scores_std
    return [], []  # Return empty for non-rank-0 processes