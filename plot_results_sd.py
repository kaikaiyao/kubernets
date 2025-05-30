import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager

def setup_plotting_style():
    """Set up the professional plotting style."""
    plt.rcParams['font.sans-serif'] = ['Segoe UI']
    plt.rcParams['font.family'] = 'sans-serif'
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 20,           # Slightly smaller for subplot layout
        'axes.labelsize': 22,      # Adjusted for subplot layout
        'axes.titlesize': 24,      # Adjusted for subplot layout
        'legend.fontsize': 18,     # Adjusted for subplot layout
        'xtick.labelsize': 20,     # Adjusted for subplot layout
        'ytick.labelsize': 20,     # Adjusted for subplot layout
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.facecolor': 'none',
        'axes.facecolor': 'white',
        'grid.linewidth': 1.0,
        'axes.linewidth': 2.0,
        'legend.title_fontsize': 20,
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'text.color': 'black',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '#E5E5E5',   # Light gray for grid
        'figure.edgecolor': 'black'
    })

# Data for Stable Diffusion models
sd_methods = [
    "SD 1.5",
    "SD 1.4",
    "SD 1.3",
    "SD 1.2",
    "SD 1.1"
]

# Updated FPR values for each model
sd_fpr_values = [
    [0.8281, 0.1953, 0.1602, 0.2536, 0.3789, 0.3984, 0.3984],  # SD 1.5
    [0.7930, 0.1172, 0.1172, 0.1883, 0.2148, 0.2344, 0.2422],  # SD 1.4
    [0.8008, 0.1914, 0.1797, 0.3016, 0.3359, 0.3359, 0.3438],  # SD 1.3
    [0.8398, 0.1914, 0.2109, 0.2586, 0.3672, 0.3633, 0.3828],  # SD 1.2
    [0.6953, 0.0195, 0.0039, 0.0768, 0.0391, 0.0625, 0.0586]   # SD 1.1
]

def create_plot(ax, methods, fpr_values):
    """Create a plot for Stable Diffusion results."""
    pixels = [4, 64, 256, 1024, 4096, 16384, 65536]
    
    # ML conference standard colorblind-friendly colors
    colors = [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE'   # Sky Blue
    ]
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    # Plot FPR vs Pixels
    for idx, method in enumerate(methods):
        marker = 'o' if idx % 2 == 0 else 's'
        label = method
        
        y_values = np.array(fpr_values[idx])
        
        # Plot main line
        line = ax.plot(range(len(pixels)), y_values,
                marker=marker, label=label, color=colors[idx],
                alpha=0.9, markersize=10, linestyle='-', linewidth=2.5)[0]
        
        lines.append(line)
        labels.append(label)
    
    # Customize the plot
    ax.set_xticks(range(len(pixels)))
    ax.set_xticklabels(pixels)
    ax.set_xlabel('Fingerprint Length (Number of Pixels Selected)', labelpad=15)
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Stable Diffusion 2.1', pad=15)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    
    # Add legend inside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
             fontsize=12, ncol=1, handletextpad=0.3, 
             handlelength=1.5, markerscale=1.2, edgecolor='black')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    return lines, labels

def create_inference_steps_comparison(ax):
    """Create a plot comparing different inference steps for 1024 pixels."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    steps_25 = [0.2536, 0.1883, 0.3016, 0.2586, 0.0768]  # 25 steps
    steps_15 = [0.1692, 0.0850, 0.1927, 0.2247, 0.0625]  # 15 steps with specific prompt
    steps_5 = [0.0000, 0.0039, 0.0039, 0.0078, 0.0000]   # 5 steps
    
    x = np.arange(len(models))
    width = 0.25
    
    # ML conference standard colors
    colors = {
        '25_steps': '#0077BB',  # Blue
        '15_steps': '#EE7733',  # Orange
        '5_steps': '#009988'    # Teal
    }
    
    # Plot bars
    bars1 = ax.bar(x - width, steps_25, width, label='25 Steps', color=colors['25_steps'])
    bars2 = ax.bar(x, steps_15, width, label='15 Steps', color=colors['15_steps'])
    bars3 = ax.bar(x + width, steps_5, width, label='5 Steps', color=colors['5_steps'])
    
    # Add value labels for all bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = val
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90,
                   fontsize=10)
    
    autolabel(bars1, steps_25)
    autolabel(bars2, steps_15)
    autolabel(bars3, steps_5)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Comparison of Inference Steps\n(1024 Pixels)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend(loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_prompt_comparison(ax):
    """Create a plot comparing prompt vs no prompt for 1024 pixels with 15 steps."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    with_prompt = [0.1692, 0.0850, 0.1927, 0.2247, 0.0625]    # 15 steps with specific prompt
    no_prompt = [0.2278, 0.2560, 0.1892, 0.4031, 0.6509]      # 15 steps with empty prompt
    
    x = np.arange(len(models))
    width = 0.35
    
    # ML conference standard colors
    colors = {
        'no_prompt': '#0077BB',    # Blue
        'with_prompt': '#EE7733'   # Orange
    }
    
    # Plot bars
    bars1 = ax.bar(x - width/2, no_prompt, width, label='Empty Prompt', color=colors['no_prompt'])
    bars2 = ax.bar(x + width/2, with_prompt, width, label='Specific Prompt', color=colors['with_prompt'])
    
    # Add value labels
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = val
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90,
                   fontsize=10)
    
    autolabel(bars1, no_prompt)
    autolabel(bars2, with_prompt)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Comparison of Prompts\n(1024 Pixels, 15 Steps)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_iteration_comparison(ax):
    """Create a plot comparing training iterations for 256 pixels."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    iter_2000 = [0.3594, 0.1992, 0.3438, 0.3906, 0.0352]  # 2000 iterations
    iter_5000 = [0.2656, 0.1523, 0.2305, 0.3008, 0.0078]  # 5000 iterations
    iter_8000 = [0.1602, 0.1172, 0.1797, 0.2109, 0.0039]  # 8000 iterations
    
    x = np.arange(len(models))
    width = 0.25  # Adjusted width to fit three bars
    
    # ML conference standard colors
    colors = {
        '2000_iter': '#0077BB',    # Blue
        '5000_iter': '#EE7733',    # Orange
        '8000_iter': '#009988'     # Teal
    }
    
    # Plot bars
    bars1 = ax.bar(x - width, iter_2000, width, label='2000 Iterations', color=colors['2000_iter'])
    bars2 = ax.bar(x, iter_5000, width, label='5000 Iterations', color=colors['5000_iter'])
    bars3 = ax.bar(x + width, iter_8000, width, label='8000 Iterations', color=colors['8000_iter'])
    
    # Add value labels
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = val
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90,
                   fontsize=10)
    
    autolabel(bars1, iter_2000)
    autolabel(bars2, iter_5000)
    autolabel(bars3, iter_8000)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Comparison of Training Iterations\n(256 Pixels)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

def create_decoder_size_comparison(ax):
    """Create a plot comparing different decoder sizes for 1024 pixels."""
    models = ["SD 1.5", "SD 1.4", "SD 1.3", "SD 1.2", "SD 1.1"]
    small_decoder = [0.9576, 0.9488, 0.9887, 0.9543, 0.9555]    # 32M params
    medium_decoder = [0.2536, 0.1883, 0.3016, 0.2586, 0.0768]   # 187M params
    large_decoder = [0.1133, 0.082, 0.1328, 0.1328, 0.0000]     # 674M params
    
    x = np.arange(len(models))
    width = 0.25
    
    # ML conference standard colors
    colors = {
        'small': '#0077BB',    # Blue
        'medium': '#EE7733',   # Orange
        'large': '#009988'     # Teal
    }
    
    # Plot bars
    bars1 = ax.bar(x - width, small_decoder, width, label='32M params', color=colors['small'])
    bars2 = ax.bar(x, medium_decoder, width, label='187M params', color=colors['medium'])
    bars3 = ax.bar(x + width, large_decoder, width, label='674M params', color=colors['large'])
    
    # Add value labels
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = val
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', rotation=90,
                   fontsize=10)
    
    autolabel(bars1, small_decoder)
    autolabel(bars2, medium_decoder)
    autolabel(bars3, large_decoder)
    
    ax.set_ylabel('FPR@95%TPR', labelpad=15)
    ax.set_title('Comparison of Decoder Sizes\n(1024 Pixels)', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(-0.05, 1.05)  # Consistent y-axis range
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

# Set up plotting style
setup_plotting_style()

# Create figure for main FPR plot
fig1 = plt.figure(figsize=(8, 6))  # More compact size
ax1 = fig1.add_subplot(111)
create_plot(ax1, sd_methods, sd_fpr_values)

# Adjust layout and save
plt.tight_layout()
plt.savefig('sd_fpr.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for inference steps comparison
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
create_inference_steps_comparison(ax2)
plt.tight_layout()
plt.savefig('sd_inference_steps_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for prompt comparison
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
create_prompt_comparison(ax3)
plt.tight_layout()
plt.savefig('sd_prompt_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for iteration comparison
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)
create_iteration_comparison(ax4)
plt.tight_layout()
plt.savefig('sd_iteration_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

# Create figure for decoder size comparison
fig5 = plt.figure(figsize=(8, 6))
ax5 = fig5.add_subplot(111)
create_decoder_size_comparison(ax5)
plt.tight_layout()
plt.savefig('sd_decoder_size_comparison.png', bbox_inches='tight', dpi=300, transparent=True)
plt.close() 