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

def train_surrogate_decoder(
    attack_type: str,
    surrogate_decoder: nn.Module,
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    epochs: int = 5, # epoch is meaningless here, consider removing
    batch_size: int = 16,
) -> None:
    """
    Train the surrogate decoder by generating batches on-the-fly on the GPU.

    Args:
        surrogate_decoder (nn.Module): The surrogate decoder model to train.
        gan_model (nn.Module): The original GAN model for generating images.
        watermarked_model (nn.Module): The watermarked GAN model.
        max_delta (float): Maximum perturbation allowed for watermarked images.
        latent_dim (int): Dimension of the latent space.
        device (torch.device): Device to perform computations on (e.g., 'cuda').
        train_size (int): Number of training samples (per class).
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Batch size for training. Defaults to 4.
    """
    time_string = generate_time_based_string()
    logging.info(f"time_string = {time_string}")

    # Move models to device and set to appropriate modes
    surrogate_decoder.train()
    surrogate_decoder.to(device)
    gan_model.eval()
    gan_model.to(device)
    watermarked_model.eval()
    watermarked_model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        surrogate_decoder.parameters(), lr=0.0001, betas=(0.5, 0.999)
    )

    # Check if GAN model is StyleGAN2
    is_stylegan2_model = is_stylegan2(gan_model)

    # Calculate number of batches (ceiling division)
    num_batches = (train_size * 2 + batch_size - 1) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx in range(num_batches):
            # Determine current batch size (handles last batch)
            current_batch_size = min(batch_size, train_size * 2 - batch_idx * batch_size)

            # Disable gradients for GAN models during batch generation
            with torch.no_grad():
                # Generate latent vectors on the device
                z = torch.randn(current_batch_size, latent_dim, device=device)
                if not is_stylegan2_model:
                    z = z.view(current_batch_size, latent_dim, 1, 1)

                # Generate images using GAN models
                if is_stylegan2_model:
                    x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                    x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
                else:
                    x_M = gan_model(z)
                    x_M_hat = watermarked_model(z)

                # Constrain the watermarked images
                x_M_hat = constrain_image(x_M_hat, x_M, max_delta)
                

            # Create labels: 0 for x_M (original), 1 for x_M_hat (watermarked)
            labels = torch.cat([
                torch.zeros(current_batch_size // 2, 1, device=device),
                torch.ones(current_batch_size // 2, 1, device=device)
            ])
            if current_batch_size % 2 != 0:
                labels = torch.cat([labels, torch.zeros(1, 1, device=device)])

            # Combine original and watermarked images
            images = torch.cat([
                x_M[:current_batch_size // 2],
                x_M_hat[current_batch_size // 2:]
            ], dim=0)
            if current_batch_size % 2 != 0:
                images = torch.cat([
                    images,
                    x_M[current_batch_size // 2:current_batch_size // 2 + 1]
                ], dim=0)

            # Shuffle images and labels
            perm = torch.randperm(current_batch_size, device=device)
            images = images[perm]
            labels = labels[perm]

            # Train on the batch
            optimizer.zero_grad()
            outputs = surrogate_decoder(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                predicted = (outputs > 0.5).float()
                correct = (predicted == labels).sum().item()
                total = labels.numel()
                batch_accuracy = correct / total

            epoch_correct += correct
            epoch_total += total

            logging.info(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Batch {batch_idx + 1}/{num_batches}, "
                f"Loss: {loss.item():.4f}, "
                f"Accuracy: {batch_accuracy:.4f}"
            )

            # Free up GPU memory
            del z, x_M, x_M_hat, images, labels, outputs, loss, predicted
            torch.cuda.empty_cache()
            gc.collect()

        # Log epoch summary
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_correct / epoch_total
        logging.info(
            f"Epoch {epoch + 1}/{epochs} Completed. "
            f"Average Loss: {avg_loss:.4f}, "
            f"Average Accuracy: {avg_accuracy:.4f}"
        )

        torch.cuda.empty_cache()
        gc.collect()

    # Save the trained model
    model_filename = f"surrogate_decoder_{time_string}.pt"
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
            image_attack_batch = torch.rand_like(x_M)
            image_attack_batches.append(image_attack_batch.cpu())

            del z, x_M, image_attack_batch
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
    attack_batch_size: int = 10,
) -> tuple:
    """
    Perform PGD attack for different alpha values.

    Args:
        surrogate_decoder (nn.Module): Trained surrogate decoder.
        decoder (nn.Module): Real decoder for scoring.
        image_attack (torch.Tensor): Images to attack.
        max_delta (float): Maximum perturbation.
        device (torch.device): Device for computations.
        num_steps (int): Number of PGD steps.
        alpha_values (list): List of step sizes to evaluate.
        attack_batch_size (int, optional): Batch size for attack. Defaults to 10.

    Returns:
        tuple: Mean and standard deviation of attack scores.
    """
    surrogate_decoder.eval()
    for param in surrogate_decoder.parameters():
        param.requires_grad = False

    criterion = nn.BCELoss()

    k_attack_scores_mean = []
    k_attack_scores_std = []

    num_attack_batches = (image_attack.size(0) + attack_batch_size - 1) // attack_batch_size

    for alpha in alpha_values:
        logging.info(f"Performing PGD attack with alpha = {alpha}")
        k_attack_scores_alpha = []

        for batch_idx in range(num_attack_batches):
            start_idx = batch_idx * attack_batch_size
            end_idx = min((batch_idx + 1) * attack_batch_size, image_attack.size(0))
            image_attack_batch = image_attack[start_idx:end_idx].clone().detach().to(device)
            image_attack_batch.requires_grad = True
            original_images = image_attack_batch.clone().detach()
            target_labels = torch.ones(image_attack_batch.size(0), 1, device=device)

            for step in range(num_steps):
                if image_attack_batch.grad is not None:
                    image_attack_batch.grad.zero_()
                outputs = surrogate_decoder(image_attack_batch)
                loss = criterion(outputs, target_labels)
                loss.backward()
                with torch.no_grad():
                    grad_sign = image_attack_batch.grad.sign()
                    image_attack_batch = image_attack_batch + alpha * grad_sign
                    image_attack_batch = torch.clamp(
                        image_attack_batch,
                        min=original_images - max_delta,
                        max=original_images + max_delta,
                    )
                    image_attack_batch.requires_grad = True

                torch.cuda.empty_cache()
                gc.collect()

            # Important Note: 
            # Here, the decoder is still based on the original key=[0] 
            # so, after the attack is finished, you get image_attack_batch, 
            # decoder[image_attack_batch] is still supposed to output 0 (if attack's successful), because the original key(2024) is [0]
            # despite the fact the surrogate decoder trains attacked image to 1 
            # (surrogate decoder's labeling as you can see in above function, labels the watermarked image (attack image goal) to 1) 
            # so, k_attack_batch -> 0 if attack is good, thus k_attack_score_batch -> 1 if attack is good, so the score is 1 if attack is good
            # so after the k_auth refactor of the code base, this still makes sense.
            with torch.no_grad():
                if attack_type in ["base_baseline"]:
                    k_attack_batch = decoder(image_attack_batch)
                elif attack_type in ["base_secure", "combined_secure", "fixed_secure"]:
                    k_mask = generate_mask_secret_key(image_attack_batch.shape, 2024, device=device) 
                    # Note that though we explicitly pass in image_attack_batch.shape, but only the channel number is needed (refer to this func), which is 3, and is always 3 across all experiments
                    k_attack_batch = decoder(mask_image_with_key(image_attack_batch, k_mask))
                else:
                    logging.error("attack_type is undefined.")
                    
            k_attack_score_batch = (
                1 - torch.norm(k_attack_batch, dim=1)
            ).cpu().numpy()
            k_attack_scores_alpha.extend(k_attack_score_batch)
            logging.info(f"Batch {batch_idx + 1}/{num_attack_batches} processed.")

            del image_attack_batch, target_labels, outputs, loss, k_attack_batch
            torch.cuda.empty_cache()
            gc.collect()

        mean_score = np.mean(k_attack_scores_alpha)
        std_score = np.std(k_attack_scores_alpha)
        k_attack_scores_mean.append(mean_score)
        k_attack_scores_std.append(std_score)
        logging.info(
            f"Alpha = {alpha}: k_attack_score mean = {mean_score:.3f}, std = {std_score:.3f}"
        )

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
    num_steps: int = 100,
    alpha_values: list = None,
) -> tuple:
    """
    Performs a label-based attack on a watermarked GAN model.

    Args:
        gan_model (nn.Module): Original GAN model.
        watermarked_model (nn.Module): Watermarked GAN model.
        max_delta (float): Maximum perturbation.
        decoder (nn.Module): Decoder for scoring.
        surrogate_decoder (nn.Module): Surrogate decoder to train.
        latent_dim (int): Latent space dimension.
        device (torch.device): Device for computations.
        train_size (int): Training samples per class.
        image_attack_size (int): Number of attack images.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        epochs (int, optional): Training epochs. Defaults to 5.
        attack_batch_size (int, optional): Batch size for attack. Defaults to 16.
        num_steps (int, optional): PGD steps. Defaults to 50.
        alpha_values (list, optional): Step sizes for PGD. Defaults to [1.0].

    Returns:
        tuple: Mean and standard deviation of attack scores.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if alpha_values is None:
        alpha_values = [1.0]

    logging.info("Starting attack_label_based function.")

    gan_model.eval()
    watermarked_model.eval()
    for param in gan_model.parameters():
        param.requires_grad = False
    for param in watermarked_model.parameters():
        param.requires_grad = False

    # Generate attack images
    image_attack = generate_attack_images(
        gan_model, image_attack_size, latent_dim, device, batch_size=10
    )
    logging.info("Initialized image_attack with random images.")

    # Train surrogate decoder
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
        batch_size=batch_size
    )
    logging.info("Training of surrogate decoder completed.")

    # Perform PGD attack
    k_attack_scores_mean, k_attack_scores_std = perform_pgd_attack(
        attack_type,
        surrogate_decoder, decoder, image_attack, 
        max_delta, device, num_steps, alpha_values, attack_batch_size
    )
    logging.info("PGD attack for different alpha values completed.")

    return k_attack_scores_mean, k_attack_scores_std