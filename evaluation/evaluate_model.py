import torch
import numpy as np
import statistics
import matplotlib.pyplot as plt
from utils.loss_functions import get_mse_loss, get_lpips_loss
from utils.image_utils import enhance_contrast, constrain_image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.transforms.functional import to_pil_image
from models.stylegan2 import is_stylegan2
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.key import generate_mask_secret_key, mask_image_with_key

def evaluate_model(
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
):
    # Set models to evaluation mode
    gan_model.eval()
    watermarked_model.eval()
    decoder.eval()
    total_decoder_params = sum(p.numel() for p in decoder.parameters())

    # Initialize loss functions
    mse_loss = get_mse_loss()
    lpips_loss = get_lpips_loss(device)

    # Initialize lists to store scores, labels, LPIPS losses, and max deltas
    scores = []
    labels = []
    loss_lpips_all = []
    max_deltas_all = []  # List to store max deltas for each image pair

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance().to(device)

    # If plotting is enabled, set up the figure and axes
    if plotting:
        rows = num_images
        cols = 3  # Original Image, Watermarked Image, Difference
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * num_images))
        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D even for single image
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i in range(num_images):
        batch_size = 1

        # Generate latent vector and images based on the GAN type
        if is_stylegan2(gan_model):
            z = torch.randn((batch_size, latent_dim), device=device)
            x_M = gan_model(z, None, truncation_psi=1.0, noise_mode="const")
            x_M_hat = watermarked_model(z, None, truncation_psi=1.0, noise_mode="const")
        else:
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            x_M = gan_model(z)
            x_M_hat = watermarked_model(z)

        # Constrain the image by max_delta
        x_M_hat = constrain_image(x_M_hat, x_M, max_delta)

        # **Compute max delta in absolute value for the image pair**
        delta = x_M_hat - x_M
        abs_delta = torch.abs(delta)
        max_delta_value = abs_delta.view(batch_size, -1).max(dim=1)[0]  # Tensor of size [batch_size]
        max_deltas_all.append(max_delta_value.item())  # Store the max delta value

        # Calculate and store LPIPS loss
        loss_lpips_all.append(lpips_loss(x_M_hat, x_M).item())
        
        # Put mask on before passing to decoder if mask_switch is on
        if mask_switch:
            if i == 0:
                k_mask = generate_mask_secret_key(x_M_hat.shape, seed_key, device=device)
        
            x_M_mask = mask_image_with_key(x_M, k_mask)
            x_M_hat_mask = mask_image_with_key(x_M_hat, k_mask)

        # Decode the images to get watermark representations
        k_M = decoder(x_M_mask)
        k_M_hat = decoder(x_M_hat_mask)

        # Calculate scores based on watermark similarity
        norm_factor = torch.sqrt(torch.tensor(len(k_auth), dtype=torch.float32)).item()
        k_M_score = 1 - (torch.norm(k_auth.unsqueeze(0) - k_M, dim=1) / norm_factor).item()
        k_M_hat_score = 1 - (torch.norm(k_auth.unsqueeze(0) - k_M_hat, dim=1) / norm_factor).item()

        # Append scores and labels
        scores.append(k_M_score)
        labels.append(0)  # Non-watermarked image label
        scores.append(k_M_hat_score)
        labels.append(1)  # Watermarked image label

        # Normalize images for FID computation
        # Assuming x_M and x_M_hat are in range [-1, 1]
        # Convert from [-1, 1] to [0, 255] and cast to uint8
        x_M_normalized = ((x_M + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        x_M_hat_normalized = ((x_M_hat + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

        # Update FID metric
        fid_metric.update(x_M_normalized, real=True)
        fid_metric.update(x_M_hat_normalized, real=False)

        # Plotting
        if plotting:
            # Original Image
            ax_orig = axes[i, 0]
            img_orig = x_M.squeeze().cpu().detach()
            img_orig = ((img_orig + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)  # Normalize from [-1,1] to [0,255]
            ax_orig.imshow(to_pil_image(img_orig))
            ax_orig.set_title(f"Original Image\nScore: {k_M_score:.4f}", fontsize=16)
            ax_orig.axis("off")

            # Watermarked Image
            ax_water = axes[i, 1]
            img_water = x_M_hat.squeeze().cpu().detach()
            img_water = ((img_water + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)  # Normalize from [-1,1] to [0,255]
            ax_water.imshow(to_pil_image(img_water))
            ax_water.set_title(f"Watermarked Image\nScore: {k_M_hat_score:.4f}", fontsize=16)
            ax_water.axis("off")

            # Difference Image
            ax_diff = axes[i, 2]
            difference_image = (x_M_hat - x_M).squeeze().cpu().detach()
            # Normalize difference image from [-2,2] to [0,255]
            difference_image = ((difference_image + 2) / 4 * 255).clamp(0, 255).to(torch.uint8)
            # difference_image = enhance_contrast(difference_image)
            ax_diff.imshow(to_pil_image(difference_image))
            ax_diff.set_title("Difference", fontsize=16)
            ax_diff.axis("off")

        # Clean up to free memory
        del z, x_M, x_M_hat, k_M, k_M_hat, k_M_score, k_M_hat_score

    # After all images are processed, compute FID
    fid_score = fid_metric.compute()

    print(plotting)
    # After all images are processed, save the figure if plotting
    if plotting:
        plt.tight_layout()
        plt.savefig("before_and_after_watermark.png")
        plt.close(fig)  # Close the figure to free memory

    # Calculate mean LPIPS loss
    loss_lpips_mean = statistics.mean(loss_lpips_all)

    # **Calculate mean of max deltas**
    mean_max_delta = statistics.mean(max_deltas_all)

    # Compute ROC AUC
    auc = roc_auc_score(labels, scores)

    # Compute ROC curve metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tnr = 1 - fpr

    # Print out all values of FPR, TPR, and thresholds explicitly
    np.set_printoptions(threshold=np.inf)  # Ensure full printing of numpy arrays
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
    
    # Calculate precision and recall
    epsilon = 1e-8  # Small value to prevent division by zero
    precision = tpr / (tpr + fpr + epsilon)
    recall = tpr

    # ### Compute TPR@1% FPR ###
    # Desired FPR threshold
    desired_fpr = 0.01  # 1%

    # Use interpolation to find the corresponding TPR
    tpr_at_1_fpr = np.interp(desired_fpr, fpr, tpr)

    # Find the threshold corresponding to desired FPR
    best_threshold = np.interp(desired_fpr, fpr, thresholds)
    best_threshold_tpr = tpr_at_1_fpr

    # Plot threshold vs TPR vs FPR
    # Generate thresholds from 0 to 1 at intervals of 0.001
    thresholds_custom = np.arange(0, 1.001, 0.001)

    # Compute TPR and FPR at each threshold
    tpr_custom = []
    fpr_custom = []

    for thresh in thresholds_custom:
        tp = sum((np.array(scores) >= thresh) & (np.array(labels) == 1))
        fp = sum((np.array(scores) >= thresh) & (np.array(labels) == 0))
        fn = sum((np.array(scores) < thresh) & (np.array(labels) == 1))
        tn = sum((np.array(scores) < thresh) & (np.array(labels) == 0))

        tpr = tp / (tp + fn + epsilon)  # True Positive Rate
        fpr = fp / (fp + tn + epsilon)  # False Positive Rate

        tpr_custom.append(tpr)
        fpr_custom.append(fpr)
    
    # Print header
    print(f"{'Thresholds':<15}{'TPR':<15}{'FPR':<15}")
    print("-" * 45)

    # Print rows with 6 decimal places
    for threshold, tpr, fpr in zip(thresholds_custom, tpr_custom, fpr_custom):
        print(f"{threshold:<15.6f}{tpr:<15.6f}{fpr:<15.6f}")

    # Plot Threshold vs TPR vs FPR
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_custom, tpr_custom, label="True Positive Rate (TPR)", color='blue', lw=2)
    plt.plot(thresholds_custom, fpr_custom, label="False Positive Rate (FPR)", color='red', lw=2)
    plt.axvline(x=best_threshold, color='green', linestyle='--', label=f"Best Threshold = {best_threshold:.3f}")
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
        best_threshold,
        best_threshold_tpr,
        loss_lpips_mean,
        fid_score,
        mean_max_delta,
        total_decoder_params,
    )
