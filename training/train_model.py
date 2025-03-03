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
import math


def train_model(
    time_string,
    gan_model,
    watermarked_model,
    decoder,
    z_classifier=None,
    n_iterations=20000,
    latent_dim=512,
    batch_size=4,
    device="cuda:0",
    run_eval=True,
    num_images=100,
    plotting=True,
    max_delta=0.01,
    saving_path="results",
    mask_switch_on=False,
    seed_key=2024,
    optimizer_M_hat=None,
    optimizer_D=None,
    start_iter=0,
    initial_loss_history=None,
    rank=0,
    world_size=1,
    key_type="csprng",
    z_dependant_training=False,
    num_classes=10,
):
    if rank == 0:
        logging.info(f"World size: {world_size}")
        logging.info(f"max_delta = {max_delta}")
        logging.info(f"time_string = {time_string}")
        logging.info("Decoder structure:\n%s", decoder.module if isinstance(decoder, DDP) else decoder)
        logging.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters())}")
        if z_dependant_training:
            logging.info(f"Using z-dependent training with {num_classes} classes")

    gan_model.eval()
    watermarked_model.train()
    decoder.train()

    loss_history = initial_loss_history if initial_loss_history is not None else []
    # Define loss function - either binary cross-entropy or cross-entropy based on mode
    if z_dependant_training:
        # Use label smoothing for better learning signal
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        if rank == 0:
            logging.info("Using CrossEntropyLoss with label smoothing for z-dependent training")
    else:
        criterion = torch.nn.BCELoss()
        
    # For z-dependent training, let's use a cyclical learning rate schedule
    if z_dependant_training:
        # Create learning rate schedulers
        def cosine_annealing_with_warmup(step, total_steps, warmup_steps=500, min_lr=1e-6, max_lr=1e-2):
            # Warmup phase
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps)) * max_lr
            # Cosine annealing phase
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        if rank == 0:
            logging.info(f"Using cosine annealing LR schedule with warmup for z-dependent training")

    for i in range(start_iter, n_iterations):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Update learning rate for z-dependent training
        if z_dependant_training:
            # Compute new learning rate
            new_lr = cosine_annealing_with_warmup(i, n_iterations)
            # Apply learning rate
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = new_lr
                
            if i % 50 == 0 and rank == 0:
                logging.info(f"Current learning rate: {new_lr:.6f}")

        # Generate different noise on each device
        torch.manual_seed(seed_key + i * world_size + rank)
        z = torch.randn((batch_size, latent_dim), device=device)

        # Get z classes if using z-dependent training
        if z_dependant_training:
            with torch.no_grad():
                z_class_logits = z_classifier(z)
                z_classes = torch.argmax(z_class_logits, dim=1)
                
                # Debugging: Print class distribution for each batch
                if rank == 0:
                    class_counts = torch.bincount(z_classes, minlength=num_classes)
                    class_str = ', '.join([f"{cls}:{count.item()}" for cls, count in enumerate(class_counts) if count > 0])
                    logging.info(f"Iteration {i} z classes: {class_str} (z shape: {z.shape})")
        
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
                k_mask = generate_mask_secret_key(x_M_hat_constrained.shape, seed_key, device=device, key_type=key_type)

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

        # Zero gradients
        optimizer_M_hat.zero_grad()
        optimizer_D.zero_grad()

        # Forward pass
        if z_dependant_training:
            # When using LUPI, we need to be careful about multiple backward passes
            # Create a fresh copy to use for multiple training iterations
            with torch.no_grad():
                x_M_hat_copy = x_M_hat_constrained.detach().clone()
            
            # Initialize optimizer for decoder
            optimizer_D.zero_grad()
            
            # Store individual losses to avoid in-place operations
            d_losses = []
            
            # Train decoder multiple times with fresh copies each time
            for train_iter in range(5):  # Train decoder 5x more than generator
                # Create a fresh computation graph for each iteration
                x_M_hat_for_decoder = x_M_hat_copy.detach().clone()
                
                # Only use watermarked images for training in z-dependent mode
                # Pass z as privileged information if the decoder supports it
                if hasattr(decoder.module, 'use_privileged_info') and decoder.module.use_privileged_info:
                    d_M_hat = decoder(x_M_hat_for_decoder, z)
                else:
                    d_M_hat = decoder(x_M_hat_for_decoder)
                
                # Compute loss using cross-entropy with z-derived classes
                d_loss = criterion(d_M_hat, z_classes)
                
                # Add L2 regularization directly
                l2_reg = 0.0
                for param in decoder.parameters():
                    l2_reg += torch.norm(param)
                d_loss = d_loss + 1e-5 * l2_reg  # Avoid in-place addition
                
                # Store each loss separately instead of accumulating in-place
                d_losses.append(d_loss)
                
                # Save loss value for logging
                if train_iter == 4:
                    d_loss_value = d_loss.item()
            
            # Combine all losses without modifying the original tensors
            # This creates a fresh computation graph connecting all losses
            combined_d_loss = sum(d_losses) / len(d_losses)
            
            # Single backward pass with combined loss
            combined_d_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            
            # Update parameters with accumulated gradients
            optimizer_D.step()
            
            # SECOND PASS - Watermarked model training
            # Create a completely fresh computation graph by detaching inputs
            with torch.no_grad():
                x_M_hat_constrained_detached = x_M_hat_constrained.detach()
            
            # Forward pass with detached inputs (after decoder optimization is complete)
            # Pass z as privileged information if the decoder supports it
            if hasattr(decoder.module, 'use_privileged_info') and decoder.module.use_privileged_info:
                d_M_hat_for_watermark = decoder(x_M_hat_constrained_detached, z)
            else:
                d_M_hat_for_watermark = decoder(x_M_hat_constrained_detached)
            
            # For better learning, use a more direct loss for the watermarked model
            # Instead of negating the CE loss, we want to maximize the probability of the correct class
            log_probs = torch.nn.functional.log_softmax(d_M_hat_for_watermark, dim=1)
            m_hat_loss = -torch.gather(log_probs, 1, z_classes.unsqueeze(1)).mean()
            
            # Add auxiliary loss to encourage diversity in outputs
            uniform_target = torch.ones_like(d_M_hat_for_watermark) / num_classes
            auxiliary_loss = torch.nn.functional.kl_div(
                log_probs, 
                uniform_target, 
                reduction='batchmean'
            ) * 0.01  # small weight
            
            m_hat_loss = m_hat_loss + auxiliary_loss
            
            # Backward pass for watermarked model
            optimizer_M_hat.zero_grad()
            m_hat_loss.backward()
            optimizer_M_hat.step()
        else:
            # Original binary classification approach
            d_M = decoder(x_M)
            d_M_hat = decoder(x_M_hat_constrained)
            
            # Target: original images -> 0, watermarked images -> 1
            zeros = torch.zeros((batch_size, 1), device=device)
            ones = torch.ones((batch_size, 1), device=device)
            
            # Compute loss
            d_loss = criterion(d_M, zeros) + criterion(d_M_hat, ones)
            
            # Backward pass for decoder
            optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            
            # Watermarked model loss - make decoder output 1 for watermarked images
            m_hat_loss = criterion(d_M_hat, ones)
            
            # Backward pass for watermarked model
            optimizer_M_hat.zero_grad()
            m_hat_loss.backward()
            optimizer_M_hat.step()

        # Record loss
        if rank == 0:
            if z_dependant_training:
                # Use the saved d_loss_value from the last decoder training iteration
                loss_history.append((d_loss_value, m_hat_loss.item()))
            else:
                loss_history.append((d_loss.item(), m_hat_loss.item()))
            
            if i % 50 == 0:
                average_d_loss = sum(l[0] for l in loss_history[-50:]) / min(50, len(loss_history[-50:]))
                average_m_hat_loss = sum(l[1] for l in loss_history[-50:]) / min(50, len(loss_history[-50:]))
                
                logging.info(f"Iteration {i}/{n_iterations}, " +
                            f"D Loss: {average_d_loss:.4f}, " +
                            f"M_hat Loss: {average_m_hat_loss:.4f}")
        
        # Save checkpoint and evaluate
        if rank == 0 and i > 0 and i % 10000 == 0:
            model_dir = os.path.join(saving_path, f"checkpoint_{i}")
            os.makedirs(model_dir, exist_ok=True)
            
            save_finetuned_model(watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                                model_dir, "watermarked_model.pth")
            save_finetuned_model(decoder.module if isinstance(decoder, DDP) else decoder,
                                model_dir, "decoder.pth")
            
            # Save loss history
            loss_hist_path = os.path.join(model_dir, "loss_history.npy")
            np.save(loss_hist_path, np.array(loss_history))
            
            if run_eval:
                with torch.no_grad():
                    if z_dependant_training:
                        evaluate_model(gan_model.module if isinstance(gan_model, DDP) else gan_model,
                                      watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                                      decoder.module if isinstance(decoder, DDP) else decoder,
                                      z_classifier=z_classifier,
                                      num_images=num_images,
                                      device=device,
                                      time_string=time_string,
                                      latent_dim=latent_dim,
                                      saving_path=saving_path,
                                      seed_key=seed_key,
                                      evaluate_from_checkpoint=True,
                                      checkpoint_iter=i,
                                      z_dependant_training=True,
                                      num_classes=num_classes)
                    else:
                        evaluate_model(gan_model.module if isinstance(gan_model, DDP) else gan_model,
                                      watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                                      decoder.module if isinstance(decoder, DDP) else decoder,
                                      num_images=num_images,
                                      device=device,
                                      time_string=time_string,
                                      latent_dim=latent_dim,
                                      saving_path=saving_path,
                                      seed_key=seed_key,
                                      evaluate_from_checkpoint=True,
                                      checkpoint_iter=i)

    # Save final model
    if rank == 0:
        model_dir = os.path.join(saving_path, "final")
        os.makedirs(model_dir, exist_ok=True)
        
        save_finetuned_model(watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                            model_dir, "watermarked_model.pth")
        save_finetuned_model(decoder.module if isinstance(decoder, DDP) else decoder,
                            model_dir, "decoder.pth")
        
        # Save loss history
        loss_hist_path = os.path.join(model_dir, "loss_history.npy")
        np.save(loss_hist_path, np.array(loss_history))
        
        # Final evaluation
        if run_eval:
            with torch.no_grad():
                if z_dependant_training:
                    evaluate_model(gan_model.module if isinstance(gan_model, DDP) else gan_model,
                                   watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                                   decoder.module if isinstance(decoder, DDP) else decoder,
                                   z_classifier=z_classifier,
                                   num_images=num_images,
                                   device=device,
                                   time_string=time_string,
                                   latent_dim=latent_dim,
                                   saving_path=saving_path,
                                   seed_key=seed_key,
                                   evaluate_from_checkpoint=True,
                                   checkpoint_iter="final",
                                   z_dependant_training=True,
                                   num_classes=num_classes)
                else:
                    evaluate_model(gan_model.module if isinstance(gan_model, DDP) else gan_model,
                                   watermarked_model.module if isinstance(watermarked_model, DDP) else watermarked_model,
                                   decoder.module if isinstance(decoder, DDP) else decoder,
                                   num_images=num_images,
                                   device=device,
                                   time_string=time_string,
                                   latent_dim=latent_dim,
                                   saving_path=saving_path,
                                   seed_key=seed_key,
                                   evaluate_from_checkpoint=True,
                                   checkpoint_iter="final")

    # Cleanup
    if rank == 0:
        logging.info("Training completed successfully.")