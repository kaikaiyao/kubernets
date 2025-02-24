#!/usr/bin/env python3
import os
import glob
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def parse_log_file(log_path):
    """
    Parses a training log file to extract iterations and corresponding loss values.
    Each log line is expected to contain something like:
      "2025-02-24 06:59:58 - Train Iteration 300000: loss_key: 0.0000, ..."
    """
    iterations = []
    losses = []
    # Regex pattern to capture the iteration number and loss value after 'loss_key:'
    pattern = re.compile(r"Train Iteration\s+(\d+):\s+loss_key:\s+([\d\.-]+)")
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iter_num = int(match.group(1))
                loss_val = float(match.group(2))
                iterations.append(iter_num)
                losses.append(loss_val)
    return iterations, losses

def average_data_by_block(iterations, losses, block_size=1000):
    """
    Averages loss values for each block of iterations.
    For example, averages losses from iterations 1 to 1000, then 1001 to 2000, etc.
    Returns:
      - averaged_iters: the average iteration index for each block (used for x-axis)
      - averaged_losses: the average loss for each block
      - max_iter_adjusted: the maximum iteration (adjusted if it's an exact multiple of block_size)
    """
    block_losses = {}
    block_iters = {}
    for it, loss in zip(iterations, losses):
        block = (it - 1) // block_size
        if block not in block_losses:
            block_losses[block] = []
            block_iters[block] = []
        block_losses[block].append(loss)
        block_iters[block].append(it)
        
    averaged_iters = []
    averaged_losses = []
    for block in sorted(block_losses.keys()):
        avg_loss = np.mean(block_losses[block])
        avg_iter = np.mean(block_iters[block])
        averaged_losses.append(avg_loss)
        averaged_iters.append(avg_iter)
        
    max_iter = iterations[-1] if iterations else 0
    # Adjust maximum iteration if it is exactly a multiple of the block_size
    max_iter_adjusted = max_iter + 1 if max_iter % block_size == 0 else max_iter
    return averaged_iters, averaged_losses, max_iter_adjusted

def extract_delta(folder_name):
    """
    Extracts the delta value from a folder name.
    For example, "delta_0_01" is interpreted as 0.01, while "delta_2" is 2.
    """
    delta_str = folder_name[len("delta_"):]
    if "_" in delta_str:
        delta_str = delta_str.replace("_", ".")
    try:
        return float(delta_str)
    except ValueError:
        return None

def plot_loss_curves(main_folder):
    # List subdirectories that start with "delta_"
    subfolders = [
        os.path.join(main_folder, d) for d in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, d)) and d.startswith("delta_")
    ]

    data_list = []  # Will store tuples of (delta, normalized x, average loss values)
    for subfolder in subfolders:
        folder_name = os.path.basename(subfolder)
        delta_val = extract_delta(folder_name)
        # Locate the training log file (using a glob pattern)
        log_files = glob.glob(os.path.join(subfolder, "training_log_*.txt"))
        if not log_files:
            print(f"[Warning] No log file found in {subfolder}. Skipping this folder.")
            continue
        # If multiple log files exist, choose the first one
        log_file = log_files[0]
        iterations, losses = parse_log_file(log_file)
        if not iterations:
            print(f"[Warning] No valid data found in {log_file}. Skipping.")
            continue
        avg_iters, avg_losses, max_iter_adj = average_data_by_block(iterations, losses, block_size=1000)
        # Normalize the averaged iteration values to [0, 1]
        x_norm = [(it - 1) / (max_iter_adj - 1) for it in avg_iters]
        data_list.append((delta_val, x_norm, avg_losses))

    # Sort the data by delta value (small to high)
    data_list.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))

    # Set up a high-quality plot style
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define a colormap to visually encode the delta value range
    delta_values = [d for d, _, _ in data_list if d is not None]
    if delta_values:
        norm = plt.Normalize(min(delta_values), max(delta_values))
        colormap = cm.viridis
    else:
        norm = None

    # Plot each delta curve with a color based on its delta value, using a thinner line (linewidth=0.8)
    for delta, x, y in data_list:
        color = colormap(norm(delta)) if (delta is not None and norm is not None) else None
        label = f"$\\delta={delta}$" if delta is not None else "Unknown"
        ax.plot(x, y, label=label, linewidth=0.8)

    # Set axis labels and title
    ax.set_xlabel("Normalized Training Progress", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title("Training Loss Over Iterations (Averaged per 1000 Iterations)", fontsize=16)
    # Only set start and end ticks for the x-axis
    ax.set_xticks([0, 1])
    ax.legend(title="Delta values", fontsize=12, title_fontsize=12)
    plt.tight_layout()

    # Save the figure in the input folder with a name like "loss_<folder_basename>.pdf"
    folder_basename = os.path.basename(os.path.normpath(main_folder))
    output_file = os.path.join(main_folder, f"loss_{folder_basename}.pdf")
    plt.savefig(output_file, format="pdf")
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curves from log files in delta subfolders (averaged per 1000 iterations).")
    parser.add_argument("folder", type=str, help="Path to the main folder containing delta subfolders")
    args = parser.parse_args()
    plot_loss_curves(args.folder)
