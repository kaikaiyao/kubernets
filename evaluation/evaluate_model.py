import torch
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
import lpips

from utils.image_utils import constrain_image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms.functional import to_pil_image
from models.stylegan2 import is_stylegan2
from torchmetrics.image.fid import FrechetInceptionDistance
from key.key import generate_mask_secret_key, mask_image_with_key

def evaluate_model(
    num_images, 
    gan_model, 
    watermarked_model, 
    decoder, 
    device,
    plotting,
    latent_dim,
    max_delta,
    mask_switch,
    seed_key,
    batch_size=8
):
    # Set models to evaluation mode
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    total_decoder_params = sum(p.numel() for p in decoder.parameters())

    # Initialize loss functions
    lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Lists to store per-image values
    scores = []
    labels = []
    loss_lpips_all = []
    max_deltas_all = []  # for max delta per image
    plot_data = []  # to store images for plotting if needed

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance().to(device)

    # Determine number of batches needed
    num_batches = math.ceil(num_images / batch_size)

    # If plotting is enabled, we will collect plotting info and then plot later.
    for batch_index in range(num_batches):
        print(f"Evaluating batch #{batch_index}")
        current_batch_size = min(batch_size, num_images - batch_index * batch_size)
        
        # Generate latent vectors and images
        if is_stylegan2(gan_model):
            z = torch.randn((current_batch_size, latent_dim), device=device)
            x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            z = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            x_M = gan_model(z)
            x_M_hat = watermarked_model(z)

        # Constrain the watermarked image based on max_delta
        x_M_hat = constrain_image(x_M_hat, x_M, max_delta)

        # Compute delta and max delta (per image)
        delta = x_M_hat - x_M
        abs_delta = torch.abs(delta)
        max_delta_values = abs_delta.view(current_batch_size, -1).max(dim=1)[0]
        for j in range(current_batch_size):
            max_deltas_all.append(max_delta_values[j].item())

        # Compute LPIPS loss (assumed to be computed per image)
        loss_lpips_batch = lpips_loss(x_M_hat, x_M)
        if loss_lpips_batch.dim() > 0:
            for j in range(current_batch_size):
                loss_lpips_all.append(loss_lpips_batch[j].item())
        else:
            for j in range(current_batch_size):
                loss_lpips_all.append(loss_lpips_batch.item())
        
        # If masking is enabled, apply secret key mask; otherwise, use the images as is.
        if mask_switch:
            if batch_index == 0:
                # Generate a key mask for the whole batch (assumes same mask works for all images)
                k_mask = generate_mask_secret_key(x_M_hat.shape, seed_key, device=device)
            x_M_mask = mask_image_with_key(x_M, k_mask)
            x_M_hat_mask = mask_image_with_key(x_M_hat, k_mask)
        else:
            x_M_mask = x_M
            x_M_hat_mask = x_M_hat

        # Decode the images to obtain watermark representations
        k_M = decoder(x_M_mask)
        k_M_hat = decoder(x_M_hat_mask)

        # Compute similarity scores based on watermark representations
        # Compute distances for each image in the batch
        k_M_scores = 1 - (torch.norm(k_M, dim=1))
        k_M_hat_scores = 1 - (torch.norm(k_M_hat, dim=1))
        for j in range(current_batch_size):
            scores.append(k_M_scores[j].item())
            labels.append(0)  # Label for non-watermarked image
            scores.append(k_M_hat_scores[j].item())
            labels.append(1)  # Label for watermarked image

        # Normalize images for FID computation (assuming range [-1, 1])
        x_M_normalized = ((x_M + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        x_M_hat_normalized = ((x_M_hat + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fid_metric.update(x_M_normalized, real=True)
        fid_metric.update(x_M_hat_normalized, real=False)

        # If plotting, store each image's data for later plotting
        if plotting:
            print("Am plotting this image")
            for j in range(current_batch_size):
                # Normalize images for plotting from [-1, 1] to [0, 255]
                img_orig = ((x_M[j] + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                img_water = ((x_M_hat[j] + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                difference_image = (x_M_hat[j] - x_M[j])
                # Normalize difference image from [-2,2] to [0,255]
                difference_image = ((difference_image + 2) / 4 * 255).clamp(0, 255).to(torch.uint8)
                plot_data.append({
                    "orig": img_orig.detach().cpu(),
                    "water": img_water.detach().cpu(),
                    "score_orig": k_M_scores[j].item(),
                    "score_water": k_M_hat_scores[j].item(),
                    "difference": difference_image.detach().cpu()
                })

        # Clean up intermediate tensors to free memory
        del z, x_M, x_M_hat, k_M, k_M_hat, k_M_scores, k_M_hat_scores

    # After processing all batches, compute FID
    print("Computing FID, this might take a while...")
    fid_score = fid_metric.compute()
    print("FID computed")

    # Plotting: plot all images in a grid if plotting is enabled
    if plotting and len(plot_data) > 0:
        print("Am plotting all images")
        total_plots = len(plot_data)
        cols = 3  # Original, Watermarked, Difference
        fig, axes = plt.subplots(nrows=total_plots, ncols=cols, figsize=(15, 5 * total_plots))
        # Ensure axes is 2D (if only one image, wrap it in a list)
        if total_plots == 1:
            axes = np.expand_dims(axes, axis=0)
        for idx, data in enumerate(plot_data):
            # Original Image
            ax_orig = axes[idx, 0]
            ax_orig.imshow(to_pil_image(data["orig"]))
            ax_orig.set_title(f"Original\nScore: {data['score_orig']:.4f}", fontsize=16)
            ax_orig.axis("off")
            # Watermarked Image
            ax_water = axes[idx, 1]
            ax_water.imshow(to_pil_image(data["water"]))
            ax_water.set_title(f"Watermarked\nScore: {data['score_water']:.4f}", fontsize=16)
            ax_water.axis("off")
            # Difference Image
            ax_diff = axes[idx, 2]
            ax_diff.imshow(to_pil_image(data["difference"]))
            ax_diff.set_title("Difference", fontsize=16)
            ax_diff.axis("off")
        plt.tight_layout()
        plt.savefig("before_and_after_watermark.png")
        plt.close(fig)

    # Compute mean LPIPS loss and mean max delta across all images
    loss_lpips_mean = statistics.mean(loss_lpips_all)
    mean_max_delta = statistics.mean(max_deltas_all)

    # Compute ROC AUC and ROC curve metrics
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Print detailed ROC metrics
    np.set_printoptions(threshold=np.inf)
    print("False Positive Rate (FPR):", fpr)
    print("True Positive Rate (TPR):", tpr)
    print("Thresholds:", thresholds)

    # Plot the ROC-AUC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", color='blue', lw=2)
    plt.plot([0, 1], [0, 1], 'r--', label="Chance Level (AUC = 0.5)")
    plt.xlabel("False Positive Rate (FPR)", fontsize=14)
    plt.ylabel("True Positive Rate (TPR)", fontsize=14)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_auc_curve.png")
    plt.close()

    # Calculate precision and recall (for reference)
    epsilon = 1e-8

    # Compute TPR@1% FPR
    desired_fpr = 0.01  # 1%
    tpr_at_1_fpr = np.interp(desired_fpr, fpr, tpr)

    # Compute TPR and FPR across a custom set of thresholds
    thresholds_custom = np.arange(0, 1.001, 0.001)
    tpr_custom = []
    fpr_custom = []
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    for thresh in thresholds_custom:
        tp = np.sum((scores_arr >= thresh) & (labels_arr == 1))
        fp = np.sum((scores_arr >= thresh) & (labels_arr == 0))
        fn = np.sum((scores_arr < thresh) & (labels_arr == 1))
        tn = np.sum((scores_arr < thresh) & (labels_arr == 0))
        current_tpr = tp / (tp + fn + epsilon)
        current_fpr = fp / (fp + tn + epsilon)
        tpr_custom.append(current_tpr)
        fpr_custom.append(current_fpr)
    
    # Print threshold, TPR, and FPR values
    print(f"{'Thresholds':<15}{'TPR':<15}{'FPR':<15}")
    print("-" * 45)
    for thresh, t, f in zip(thresholds_custom, tpr_custom, fpr_custom):
        print(f"{thresh:<15.6f}{t:<15.6f}{f:<15.6f}")

    # Plot Threshold vs TPR vs FPR
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_custom, tpr_custom, label="True Positive Rate (TPR)", color='blue', lw=2)
    plt.plot(thresholds_custom, fpr_custom, label="False Positive Rate (FPR)", color='red', lw=2)
    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Rate", fontsize=14)
    plt.title("Threshold vs TPR vs FPR", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("threshold_vs_tpr_vs_fpr.png")
    plt.close()

    return (
        auc,
        tpr_at_1_fpr,
        loss_lpips_mean,
        fid_score,
        mean_max_delta,
        total_decoder_params,
    )
