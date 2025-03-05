import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import seaborn as sns

def parse_log_file(filename):
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Train Iteration' in line:
                # Extract loss value using regex
                match = re.search(r'loss: (\d+\.\d+)', line)
                if match:
                    losses.append(float(match.group(1)))
    return np.array(losses)

def smooth_losses(losses, window_size=100):
    # If we have fewer points than window_size, adjust window_size
    if len(losses) < window_size:
        window_size = max(1, len(losses) // 10)
    
    # Pad the array to handle edge cases
    pad_size = window_size - (len(losses) % window_size)
    if pad_size < window_size:
        padded_losses = np.pad(losses, (0, pad_size), mode='edge')
    else:
        padded_losses = losses
    
    # Reshape and compute means
    n_windows = len(padded_losses) // window_size
    reshaped = padded_losses[:n_windows * window_size].reshape(-1, window_size)
    return np.mean(reshaped, axis=1)

def plot_loss(filename, output_path='loss_plot.pdf'):
    # Set the style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("deep")
    
    # Create figure with high DPI for quality
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Parse and smooth the losses
    losses = parse_log_file(filename)
    smoothed_losses = smooth_losses(losses)
    
    # Create x-axis points
    x = np.linspace(0, 1, len(smoothed_losses))
    
    # Plot with professional styling
    plt.plot(x, smoothed_losses, linewidth=2.5, color='#2171b5')
    
    # Customize the plot
    plt.xlabel('Training Progress', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    
    # Set x-axis ticks to only show 0 and 1
    plt.xticks([0, 1], ['0', '1'])
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "loss1.txt"  # default filename
    plot_loss(log_file)
    print(f"Plot saved as loss_plot.pdf") 