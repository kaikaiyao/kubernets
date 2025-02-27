import torch
import os
import gc
import numpy as np
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

from models.model_utils import save_finetuned_model
from evaluation.evaluate_model import evaluate_model
from models.stylegan2 import is_stylegan2
from utils.image_utils import constrain_image
from key.key import generate_mask_secret_key, mask_image_with_key

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def train_model(
    time_string,
    gan_model,
    watermarked_model,
    decoder,
    n_iterations,
    latent_dim,
    batch_size,
    device,
    run_eval,
    num_images,
    plotting,
    max_delta,
    saving_path,
    mask_switch_on,
    seed_key,
    optimizer_M_hat,
    optimizer_D,
    start_iter=0,
    initial_loss_history=None,
    rank=0,
    world_size=1,
):
    if rank == 0:
        logging.info(f"World size: {world_size}")
        logging.info(f"max_delta = {max_delta}")
        logging.info(f"time_string = {time_string}")
        logging.info("Decoder structure:\n%s", decoder.module if isinstance(decoder, DDP) else decoder)
        logging.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")

    gan_model.eval()
    watermarked_model.train()
    decoder.train()

    loss_history = initial_loss_history if initial_loss_history is not None else []

    for i in range(start_iter, n_iterations):
        torch.cuda.empty_cache()
        gc.collect()

        # Generate different noise on each device
        torch.manual_seed(seed_key + i * world_size + rank)
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

        if mask_switch_on:
            if i == 0 or start_iter != 0:
                k_mask = generate_mask_secret_key(x_M_hat_constrained.shape, seed_key, device=device)

            x_M_original = x_M.clone().detach()
            x_M_hat_constrained_original = x_M_hat_constrained.clone().detach()

            x_M = mask_image_with_key(x_M, k_mask)
            x_M_hat_constrained = mask_image_with_key(x_M_hat_constrained, k_mask)

            if plotting and rank == 0 and (i == 0 or i == 99999):
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

        d_k_M_hat = torch.norm(k_M_hat, dim=1) 
        d_k_M = torch.norm(k_M, dim=1) 

        del k_M, k_M_hat
        torch.cuda.empty_cache()

        loss = ((d_k_M - d_k_M_hat).max() + 1) ** 2 

        optimizer_M_hat.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)

        loss.backward()

        optimizer_M_hat.step()
        optimizer_D.step()

        loss_history.append(loss.item())

        if rank == 0:
            logging.info(
                f"Train Iteration {i + 1}: "
                f"loss: {loss.item():.4f}, "
                f"d_k_M_hat.max(): {d_k_M_hat.max().item():.4f}, "
                f"d_k_M_hat.min(): {d_k_M_hat.min().item():.4f}, "
                f"d_k_M.max(): {d_k_M.max().item():.4f}"
                f"d_k_M.min(): {d_k_M.min().item():.4f}"
            )

        del loss, d_k_M_hat, d_k_M
        torch.cuda.empty_cache()

    # Save final models
    if rank == 0:
        checkpoint = {
            'watermarked_model': watermarked_model.module.state_dict(),
            'decoder': decoder.module.state_dict(),
            'optimizer_M_hat': optimizer_M_hat.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'iteration': n_iterations - 1,
            'loss_history': loss_history,
        }
        checkpoint_path = os.path.join(saving_path, f'checkpoint_final_{time_string}.pt')
        torch.save(checkpoint, checkpoint_path)

        save_finetuned_model(watermarked_model.module, saving_path, f'watermarked_model_final_{time_string}.pkl')
        torch.save(decoder.module.state_dict(), os.path.join(saving_path, f'decoder_model_final_{time_string}.pth'))
        logging.info(f"Final models saved at iteration {n_iterations}, time_string = {time_string}")

    # Final evaluation and logging
    if rank == 0:
        logging.info("Training completed.")
        
        if run_eval:
            watermarked_model.eval()
            decoder.eval()
            
            with torch.no_grad():
                eval_results = evaluate_model(
                    num_images,
                    gan_model,
                    watermarked_model.module,
                    decoder.module,
                    device,
                    plotting,
                    latent_dim,
                    max_delta,
                    mask_switch_on,
                    seed_key,
                    flip_key_type="none",
                )
            auc, tpr_at_1_fpr, lpips_loss, fid_score, mean_max_delta, total_decoder_params = eval_results
        
            auc_str = f"{auc:.4f}" if auc is not None else "None"
            tpr_str = f"{tpr_at_1_fpr:.4f}" if tpr_at_1_fpr is not None else "None"

            logging.info(
                f"Final evaluation after {n_iterations} iterations: "
                f"AUC score: {auc_str}, "
                f"tpr_at_1_fpr: {tpr_str}, "
                f"lpips_loss: {lpips_loss:.4f}, "
                f"fid_score: {fid_score:.4f}, "
                f"mean_max_delta: {mean_max_delta:.4f}, "
                f"total_decoder_params: {total_decoder_params}"
            )

    # Cleanup
    if rank == 0:
        logging.info("Training completed successfully.")