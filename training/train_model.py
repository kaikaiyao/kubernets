import torch
from torch import optim
import os
import gc
import numpy as np
import logging  # Added logging import

from utils.model_utils import save_finetuned_model
from utils.loss_functions import get_key_loss
from evaluation.evaluate_model import evaluate_model
from utils.file_utils import generate_time_based_string
from models.stylegan2 import is_stylegan2
from utils.image_utils import constrain_image
from utils.key import generate_mask_secret_key, mask_image_with_key
from utils.logging import setup_logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def train_model(
    gan_model,
    watermarked_model,
    decoder,
    k_auth,
    n_iterations,
    latent_dim,
    batch_size,
    device,
    lr_M_hat,
    lr_D,
    run_eval,
    num_images,
    plotting,
    max_delta,
    saving_path,
    convergence_threshold,
    mask_switch,
    seed_key,
    mask_threshold,
):
    # Generate time_string early for logging
    time_string = generate_time_based_string()
    
    # Ensure saving directory exists
    os.makedirs(saving_path, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(saving_path, f'training_log_{time_string}.txt')
    setup_logging(log_file)  # Call the logging setup function
    
    # Print max_delta (might need a better restructure of logging code altogether later)
    logging.info(f"max_delta = {max_delta}")

    # Log initial configuration
    logging.info(f"time_string = {time_string}")
    logging.info("The decoder structure is:\n%s", decoder)
    logging.info(f"The decoder model's number of parameters is: {sum(p.numel() for p in decoder.parameters())}")
    logging.info(f"The convergence threshold is: {convergence_threshold}")

    # Modified multi-GPU setup
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        # Ensure models are on correct device before wrapping
        watermarked_model = watermarked_model.to(device)
        decoder = decoder.to(device)
        watermarked_model = torch.nn.DataParallel(watermarked_model)
        decoder = torch.nn.DataParallel(decoder)

    optimizer_D = optim.Adagrad(decoder.parameters(), lr=lr_D)
    optimizer_M_hat = optim.Adagrad(watermarked_model.parameters(), lr=lr_M_hat)

    gan_model.eval()
    watermarked_model.train()
    decoder.train()

    for param in watermarked_model.parameters():
        param.requires_grad = True

    loss_key_history = []
    converged = False

    for i in range(n_iterations):
        torch.cuda.empty_cache()
        gc.collect()

        z = torch.randn((batch_size, latent_dim), device=device)

        with torch.no_grad():
            if is_stylegan2(gan_model):
                x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            else:
                x_M = gan_model(z)

        if is_stylegan2(gan_model):
            x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            x_M_hat = watermarked_model(z)
        
        x_M_hat_constrained = constrain_image(x_M_hat, x_M, max_delta)

        if mask_switch:
            if i == 0:
                k_mask = generate_mask_secret_key(x_M_hat_constrained.shape, seed_key, device=device)

            x_M_original = x_M.clone().detach()
            x_M_hat_constrained_original = x_M_hat_constrained.clone().detach()

            x_M = mask_image_with_key(x_M, k_mask)
            x_M_hat_constrained = mask_image_with_key(x_M_hat_constrained, k_mask)

            if i == 0 or i == 99999:
                os.makedirs(saving_path, exist_ok=True)
                with PdfPages(os.path.join(saving_path, 'first_iteration_images.pdf')) as pdf:
                    fig, axes = plt.subplots(4, 8, figsize=(20, 12))
                    fig.suptitle("First Iteration Images", fontsize=16)

                    for idx in range(8):
                        def normalize_image(img):
                            img = img.cpu().detach().numpy()
                            img_min = img.min()
                            img_max = img.max()
                            img = (img - img_min) / (img_max - img_min) * 255.0
                            return img.astype('uint8')

                        axes[0, idx].imshow(normalize_image(x_M_original[idx].permute(1, 2, 0)))
                        axes[0, idx].set_title(f"Original x_M {idx+1}")
                        axes[0, idx].axis('off')

                        axes[1, idx].imshow(normalize_image(x_M[idx].permute(1, 2, 0)))
                        axes[1, idx].set_title(f"Modified x_M {idx+1}")
                        axes[1, idx].axis('off')

                        axes[2, idx].imshow(normalize_image(x_M_hat_constrained_original[idx].permute(1, 2, 0)))
                        axes[2, idx].set_title(f"Original x_M_hat {idx+1}")
                        axes[2, idx].axis('off')

                        axes[3, idx].imshow(normalize_image(x_M_hat_constrained[idx].permute(1, 2, 0)))
                        axes[3, idx].set_title(f"Modified x_M_hat {idx+1}")
                        axes[3, idx].axis('off')

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        x_M = x_M.detach()

        k_M = decoder(x_M)
        k_M_hat = decoder(x_M_hat_constrained)

        del x_M, x_M_hat, x_M_hat_constrained
        torch.cuda.empty_cache()

        d_k_M_hat = torch.norm(k_auth.unsqueeze(0).expand(batch_size, -1) - k_M_hat, dim=1) / torch.sqrt(torch.tensor(len(k_auth), dtype=torch.float32, device=device))
        d_k_M = torch.norm(k_auth.unsqueeze(0).expand(batch_size, -1) - k_M, dim=1) / torch.sqrt(torch.tensor(len(k_auth), dtype=torch.float32, device=device))

        del k_M, k_M_hat
        torch.cuda.empty_cache()

        loss_key = get_key_loss(d_k_M_hat, d_k_M)

        optimizer_M_hat.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)

        loss = loss_key
        loss.backward()

        optimizer_M_hat.step()
        optimizer_D.step()

        loss_key_history.append(loss_key.item())

        logging.info(
            f"Train Iteration {i + 1}: "
            f"loss_key: {loss_key.item():.4f}, "
            f"d_k_M_hat.max(): {d_k_M_hat.max().item():.4f}, "
            f"d_k_M.min(): {d_k_M.min().item():.4f}"
        )

        del loss, loss_key, d_k_M_hat, d_k_M
        torch.cuda.empty_cache()

        if (i + 1) % 2000 == 0 and (i + 1) >= 4000:
            current_avg_loss_key = np.mean(loss_key_history[-2000:])
            previous_avg_loss_key = np.mean(loss_key_history[-4000:-2000])
            loss_key_diff = abs(current_avg_loss_key - previous_avg_loss_key)
            
            logging.info(f"At iteration {i + 1}, loss_key difference between last two 2000-iteration blocks: {loss_key_diff:.4f}")
            
            if loss_key_diff < convergence_threshold:
                converged = True
                logging.info(f"Converged at iteration {i + 1}, loss_key difference: {loss_key_diff:.4f}")
                
                if run_eval:
                    watermarked_model.eval()
                    decoder.eval()
                    
                    with torch.no_grad():
                        eval_results = evaluate_model(
                            num_images,
                            gan_model,
                            watermarked_model,
                            decoder,
                            k_auth,
                            device,
                            plotting,
                            latent_dim,
                            max_delta,
                            mask_switch,
                            seed_key,
                            mask_threshold,
                        )
                        auc, tpr_at_1_fpr, best_threshold, best_threshold_tpr, loss_lpips_mean, fid_score, mean_max_delta, total_decoder_params = eval_results
                    
                    logging.info(
                        f"Eval after convergence at iteration {i + 1}: "
                        f"AUC score: {auc:.4f if auc is not None else 'None'}, "
                        f"tpr_at_1_fpr: {tpr_at_1_fpr:.4f if tpr_at_1_fpr is not None else 'None'}, "
                        f"best_threshold: {best_threshold:.4f if best_threshold is not None else 'None'}, "
                        f"best_threshold_tpr: {best_threshold_tpr:.4f if best_threshold_tpr is not None else 'None'}, "
                        f"loss_lpips_mean: {loss_lpips_mean:.4f}, "
                        f"fid_score: {fid_score:.4f}, "
                        f"mean_max_delta: {mean_max_delta:.4f}, "
                        f"total_decoder_params: {total_decoder_params}"
                    )
                    
                    # Save the underlying model if using DataParallel
                    watermarked_model_to_save = watermarked_model.module if isinstance(watermarked_model, torch.nn.DataParallel) else watermarked_model
                    save_finetuned_model(watermarked_model_to_save, saving_path, f'watermarked_model_{time_string}.pkl')
                    torch.save(decoder.state_dict(), os.path.join(saving_path, f'decoder_model_{time_string}.pth'))
                    logging.info(f"Models saved after convergence at iteration {i + 1}, time_string = {time_string}")
                break

    if not converged:
        convergence_score = None
        if len(loss_key_history) >= 4000:
            current_avg_loss_key = np.mean(loss_key_history[-2000:])
            previous_avg_loss_key = np.mean(loss_key_history[-4000:-2000])
            convergence_score = abs(current_avg_loss_key - previous_avg_loss_key)
        
        logging.info(
            f"Training completed. Convergence score: {convergence_score:.4f}" 
            if convergence_score is not None 
            else "Training completed. Convergence score: Not enough data to compute convergence score"
        )
        
        if run_eval:
            watermarked_model.eval()
            decoder.eval()
            
            with torch.no_grad():
                eval_results = evaluate_model(
                    num_images,
                    gan_model,
                    watermarked_model,
                    decoder,
                    k_auth,
                    device,
                    plotting,
                    latent_dim,
                    max_delta,
                    mask_switch,
                    seed_key,
                    mask_threshold,
                )
                auc, tpr_at_1_fpr, best_threshold, best_threshold_tpr, loss_lpips_mean, fid_score, mean_max_delta, total_decoder_params = eval_results
            
            logging.info(
                f"Eval after training completion at iteration {n_iterations}: "
                f"AUC score: {f'{auc:.4f}' if auc is not None else 'None'}, "
                f"tpr_at_1_fpr: {f'{tpr_at_1_fpr:.4f}' if tpr_at_1_fpr is not None else 'None'}, "
                f"best_threshold: {f'{best_threshold:.4f}' if best_threshold is not None else 'None'}, "
                f"best_threshold_tpr: {f'{best_threshold_tpr:.4f}' if best_threshold_tpr is not None else 'None'}, "
                f"loss_lpips_mean: {loss_lpips_mean:.4f}, "
                f"fid_score: {fid_score:.4f}, "
                f"mean_max_delta: {mean_max_delta:.4f}, "
                f"total_decoder_params: {total_decoder_params}"
            )
            
            # Save the underlying model if using DataParallel
            watermarked_model_to_save = watermarked_model.module if isinstance(watermarked_model, torch.nn.DataParallel) else watermarked_model
            save_finetuned_model(watermarked_model_to_save, saving_path, f'watermarked_model_{time_string}.pkl')
            torch.save(decoder.state_dict(), os.path.join(saving_path, f'decoder_model_{time_string}.pth'))
            logging.info(f"Models saved after training completion, time_string = {time_string}")