import os
import gc
import logging
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.stylegan2 import is_stylegan2
from utils.image_utils import constrain_image
from utils.file_utils import generate_time_based_string
from utils.key import generate_mask_secret_key, mask_image_with_key

class GANGeneratedDataset(Dataset):
    """
    Custom Dataset for generating images and labels on-the-fly for training the surrogate decoder.
    """
    def __init__(
        self,
        gan_model: nn.Module,
        watermarked_model: nn.Module,
        max_delta: float,
        latent_dim: int,
        device: torch.device,
        train_size: int,
    ):
        # Keep models on the device (GPU)
        self.gan_model = gan_model.to(device)
        self.watermarked_model = watermarked_model.to(device)
        self.max_delta = max_delta
        self.latent_dim = latent_dim
        self.device = device
        self.train_size = train_size
        self.is_stylegan2 = is_stylegan2(gan_model)

    def __len__(self):
        return self.train_size * 2  # Since we have x_M and x_M_hat

    def __getitem__(self, idx):
        with torch.no_grad():
            # Generate latent vector on the device
            z = torch.randn(1, self.latent_dim, device=self.device)
            if not self.is_stylegan2:
                z = z.view(1, self.latent_dim, 1, 1)

            # Generate images on the device
            if self.is_stylegan2:
                x_M = self.gan_model(z, None, truncation_psi=1.0, noise_mode="const")
                x_M_hat = self.watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                x_M = self.gan_model(z)
                x_M_hat = self.watermarked_model(z)

            # Constrain the watermarked image
            x_M_hat = constrain_image(x_M_hat, x_M, self.max_delta)

            # # Put mask on before passing to decoder if mask_switch is on
            # k_mask = generate_mask_secret_key(x_M_hat.shape, 2025, device='cuda')
            # x_M = mask_image_with_key(x_M, k_mask, 0.2)
            # x_M_hat = mask_image_with_key(x_M_hat, k_mask, 0.2)

            # Move to CPU and detach
            x_M = x_M.squeeze(0).cpu()
            x_M_hat = x_M_hat.squeeze(0).cpu()

            # Alternate between x_M and x_M_hat
            if idx % 2 == 0:
                return x_M, torch.tensor([0], dtype=torch.float32)
            else:
                return x_M_hat, torch.tensor([1], dtype=torch.float32)

def train_surrogate_decoder(
    surrogate_decoder: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 4,
) -> None:
    """
    Train the surrogate decoder.
    """

    time_string = generate_time_based_string()
    print(f"time_string = {time_string}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Ensure data loading happens in main process
        pin_memory=True,
    )

    surrogate_decoder.train()
    surrogate_decoder.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        surrogate_decoder.parameters(), lr=0.0001, betas=(0.5, 0.999)
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
            # Move data to the appropriate device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = surrogate_decoder(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy for the current batch
            with torch.no_grad():
                predicted = (outputs > 0.5).float()
                correct = (predicted == batch_labels).sum().item()
                total = batch_labels.numel()
                batch_accuracy = correct / total

            epoch_correct += correct
            epoch_total += total

            logging.info(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, "
                f"Accuracy: {batch_accuracy:.4f}"
            )

            # Free up GPU memory after each batch
            del batch_images, batch_labels, outputs, loss, predicted
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = epoch_correct / epoch_total

        logging.info(
            f"Epoch {epoch + 1}/{epochs} Completed. "
            f"Average Loss: {avg_loss:.4f}, "
            f"Average Accuracy: {avg_accuracy:.4f}"
        )

        # Free up GPU memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save the surrogate_decoder after the final epoch
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
    """
    image_attack_batches = []
    num_batches = math.ceil(image_attack_size / batch_size)

    # Keep GAN model on the device (GPU)
    gan_model.to(device)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, image_attack_size - batch_idx * batch_size)
            # Generate images on the device
            if is_stylegan2(gan_model):
                z = torch.randn((current_batch_size, latent_dim), device=device)
                x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
                x_M = gan_model(z)
            # Attack image construction 1: Generate random image for attack and append to list
            image_attack_batch = torch.rand_like(x_M)
            # # Attack image construction 2: Use x_M directly
            # image_attack_batch = x_M
            image_attack_batches.append(image_attack_batch.cpu())

            # Clean up
            del z, x_M, image_attack_batch
            torch.cuda.empty_cache()
            gc.collect()

    # Concatenate batches and move to the target device
    image_attack = torch.cat(image_attack_batches).to(device)
    del image_attack_batches
    torch.cuda.empty_cache()
    gc.collect()

    return image_attack

def perform_pgd_attack(
    surrogate_decoder: nn.Module,
    decoder: nn.Module,
    image_attack: torch.Tensor,
    k_auth: torch.Tensor,
    best_threshold: float,
    max_delta: float,
    device: torch.device,
    num_steps: int,
    alpha_values: list,
    attack_batch_size: int = 10,
) -> tuple:
    """
    Perform PGD attack for different alpha values.
    """
    # Set surrogate_decoder to evaluation mode and disable gradients
    surrogate_decoder.eval()
    for param in surrogate_decoder.parameters():
        param.requires_grad = False

    criterion = nn.BCELoss()

    k_attack_scores_mean = []
    k_attack_scores_std = []
    success_rates = []

    num_attack_batches = (
        image_attack.size(0) + attack_batch_size - 1
    ) // attack_batch_size

    for alpha in alpha_values:
        logging.info(f"Performing PGD attack with alpha = {alpha}")
        k_attack_scores_alpha = []

        for batch_idx in range(num_attack_batches):
            start_idx = batch_idx * attack_batch_size
            end_idx = min(
                (batch_idx + 1) * attack_batch_size, image_attack.size(0)
            )
            image_attack_batch = (
                image_attack[start_idx:end_idx]
                .clone()
                .detach()
                .to(device)
            )
            image_attack_batch.requires_grad = True
            original_images = image_attack_batch.clone().detach()
            target_labels = torch.ones(
                image_attack_batch.size(0), 1, dtype=torch.float32, device=device
            )

            for step in range(num_steps):
                # Zero the gradients
                if image_attack_batch.grad is not None:
                    image_attack_batch.grad.zero_()

                # Forward Pass
                outputs = surrogate_decoder(image_attack_batch)
                loss = criterion(outputs, target_labels)

                # Backward Pass
                loss.backward()

                # Gradient Ascent Step
                with torch.no_grad():
                    grad_sign = image_attack_batch.grad.sign()
                    image_attack_batch = image_attack_batch + alpha * grad_sign
                    image_attack_batch = torch.clamp(
                        image_attack_batch,
                        min=original_images - max_delta,
                        max=original_images + max_delta,
                    )
                    image_attack_batch.requires_grad = True  # Ensure gradients are re-enabled after modification

                # Free up GPU memory after each PGD step
                torch.cuda.empty_cache()
                gc.collect()

            # temporary: mask the images
            k_mask = generate_mask_secret_key(image_attack_batch.shape, 2024, device=device)
            image_attack_batch = mask_image_with_key(image_attack_batch, k_mask)
            
            # Compute k_attack_score for the batch using real decoder
            with torch.no_grad():
                k_attack_batch = decoder(image_attack_batch)

            norm_factor = torch.sqrt(torch.tensor(len(k_auth), dtype=torch.float32))
            k_attack_score_batch = (
                1
                - torch.norm(
                    k_auth.unsqueeze(0) - k_attack_batch, dim=1
                )
                / norm_factor
            ).cpu().numpy()
            k_attack_scores_alpha.extend(k_attack_score_batch)
            success_rate_batch = np.mean(k_attack_score_batch > best_threshold)
            logging.info(
                f"Batch {batch_idx + 1}/{num_attack_batches}: Success rate = {success_rate_batch:.3f}"
            )

            # Free up GPU memory after each batch
            del image_attack_batch, target_labels, outputs, loss, k_attack_batch
            torch.cuda.empty_cache()
            gc.collect()

        # Set the print options to ensure all elements are displayed
        np.set_printoptions(threshold=np.inf)
        print(f"k_attack_scores_alpha: {k_attack_scores_alpha}")

        mean_score = np.mean(k_attack_scores_alpha)
        std_score = np.std(k_attack_scores_alpha)
        success_rate = np.mean(
            np.array(k_attack_scores_alpha) > best_threshold
        )
        k_attack_scores_mean.append(mean_score)
        k_attack_scores_std.append(std_score)
        success_rates.append(success_rate)
        logging.info(
            f"Alpha = {alpha}: k_attack_score mean = {mean_score:.3f}, std = {std_score:.3f}, success_rate = {success_rate:.3f}"
        )

    return k_attack_scores_mean, k_attack_scores_std, success_rates

def plot_k_attack_scores(
    alpha_values: list,
    k_attack_scores_mean: list,
    k_attack_scores_std: list,
    output_dir: str,
) -> None:
    """
    Plot Mean and Std of k_attack_score vs Alpha.
    """
    # Set seaborn style
    sns.set(style="whitegrid", context="notebook")

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'alpha': alpha_values,
        'Mean_k_attack_score': k_attack_scores_mean,
        'Std_k_attack_score': k_attack_scores_std
    })

    # Sort the DataFrame by alpha
    df_plot = df_plot.sort_values('alpha').reset_index(drop=True)

    # Initialize the plot
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(
        x='alpha',
        y='Mean_k_attack_score',
        data=df_plot,
        marker='o',
        label='Mean $k_{attack\\_score}$'
    )

    # Add shaded area for standard deviation
    ax.fill_between(
        df_plot['alpha'],
        df_plot['Mean_k_attack_score'] - df_plot['Std_k_attack_score'],
        df_plot['Mean_k_attack_score'] + df_plot['Std_k_attack_score'],
        alpha=0.2,
        label='Std Dev'
    )

    # Set title and labels
    ax.set_title('$k_{attack\\_score}$ vs Alpha')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Mean $k_{attack\\_score}$')
    ax.set_ylim(0.95, 1.05)
    ax.legend()

    # Optimize layout
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the filename
    plot_filename = os.path.join(output_dir, 'mean_std_k_attack_score_vs_alpha.pdf')

    # Save the figure
    plt.savefig(plot_filename)
    logging.info(f"Plot saved as {plot_filename}")

    plt.close()

def black_box_attack_binary_based(
    gan_model: nn.Module,
    watermarked_model: nn.Module,
    max_delta: float,
    decoder: nn.Module,
    surrogate_decoder: nn.Module,
    k_auth: torch.Tensor,
    latent_dim: int,
    device: torch.device,
    train_size: int,
    image_attack_size: int,
    best_threshold: float,
    batch_size: int = 16,
    epochs: int = 5,
    attack_batch_size: int = 16,
    num_steps: int = 50,
    alpha_values: list = None,
    output_dir: str = '.',
) -> tuple:
    """
    Performs a black-box attack on a watermarked GAN model using a surrogate decoder.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if alpha_values is None:
        alpha_values = [1.0]

    logging.info("Starting black_box_attack_binary_based function.")

    # Set models to evaluation mode and disable gradients
    gan_model.eval()
    watermarked_model.eval()
    for param in gan_model.parameters():
        param.requires_grad = False
    for param in watermarked_model.parameters():
        param.requires_grad = False

    # Generate images for attack
    image_attack = generate_attack_images(
        gan_model, image_attack_size, latent_dim, device, batch_size=10
    )
    logging.info("Initialized image_attack with random images.")

    # Create Dataset for on-the-fly data generation
    dataset = GANGeneratedDataset(
        gan_model, watermarked_model, max_delta, latent_dim, device, train_size
    )
    logging.info("GANGeneratedDataset created for on-the-fly data generation.")

    # Train the Surrogate Decoder
    train_surrogate_decoder(
        surrogate_decoder, dataset, device, epochs, batch_size
    )
    logging.info("Training of surrogate decoder completed.")

    # Perform PGD Attack
    k_attack_scores_mean, k_attack_scores_std, success_rates = perform_pgd_attack(
        surrogate_decoder, decoder, image_attack, k_auth, best_threshold,
        max_delta, device, num_steps, alpha_values, attack_batch_size
    )
    logging.info("PGD attack for different alpha values completed.")

    # # Plot results (not plotting for now)
    # plot_k_attack_scores(
    #     alpha_values, k_attack_scores_mean, k_attack_scores_std, output_dir
    # )
    # logging.info("Plotting completed.")
    # logging.info("black_box_attack_binary_based function completed.")

    return k_attack_scores_mean, k_attack_scores_std, success_rates
