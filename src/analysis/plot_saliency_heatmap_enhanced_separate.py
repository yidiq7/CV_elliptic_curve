import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_enhanced_heatmap(size, map_data, title, filename_suffix, cmap='hot'):
    folder = f'saliency_maps_{size}'
    os.makedirs(folder, exist_ok=True)
    
    # We want to show standard deviation across columns (twists) for each row (prime)
    std_across_twists = map_data.std(axis=1)
    # Use max instead of mean to preserve the sparse mathematical signal
    max_across_twists = map_data.max(axis=1)
    
    # We want to show standard deviation across rows (primes) for each column (twist)
    std_across_primes = map_data.std(axis=0)
    # Use max instead of mean to preserve the sparse mathematical signal
    max_across_primes = map_data.max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the 2D heatmap
    # Let's clip to a high percentile to prevent a single pixel from washing out the colormap
    vmax = np.percentile(np.abs(map_data), 99.5)
    vmin = -vmax if cmap == 'coolwarm' else 0
    
    im = ax.imshow(map_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'{title} (N={size})', fontsize=14)
    ax.set_ylabel('Primes ($p$)', fontsize=12)
    ax.set_xlabel('Twists ($\chi$)', fontsize=12)

    # Create dividers for marginal plots
    divider = make_axes_locatable(ax)
    
    # Marginal plot for Primes (Right side) - Projection onto Primes
    ax_prime = divider.append_axes("right", size="20%", pad=0.1)
    ax_prime.plot(max_across_twists, range(size), color='red', label='Max Saliency')
    ax_prime.fill_betweenx(range(size), 
                           max_across_twists - std_across_twists, 
                           max_across_twists + std_across_twists, 
                           color='red', alpha=0.2)
    ax_prime.invert_yaxis()  # Match image coordinates
    ax_prime.set_xlabel('Max Abs Grad', fontsize=10)
    ax_prime.set_yticks([])
    ax_prime.grid(True, alpha=0.3)
    ax_prime.legend(loc='upper right', fontsize=8)
    
    # Marginal plot for Twists (Top side) - Projection onto Twists
    ax_twist = divider.append_axes("top", size="20%", pad=0.1)
    ax_twist.plot(range(size), max_across_primes, color='blue', label='Max Saliency')
    ax_twist.fill_between(range(size), 
                          max_across_primes - std_across_primes, 
                          max_across_primes + std_across_primes, 
                          color='blue', alpha=0.2)
    ax_twist.set_ylabel('Max Abs Grad', fontsize=10)
    ax_twist.set_xticks([])
    ax_twist.grid(True, alpha=0.3)
    ax_twist.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, f'enhanced_marginal_{filename_suffix}_{size}.png'), dpi=300)
    plt.close()

def process_folder(size):
    folder = f'saliency_maps_{size}'
    rank0_path = os.path.join(DATA_DIR, folder, 'rank0_saliency_avg.npy')
    rank1_path = os.path.join(DATA_DIR, folder, 'rank1_saliency_avg.npy')
    fake_path = os.path.join(DATA_DIR, folder, 'fake_saliency_avg.npy')
    
    if not (os.path.exists(rank0_path) and os.path.exists(fake_path)):
        return

    r0_map = np.load(rank0_path).mean(axis=0)
    r1_map = np.load(rank1_path).mean(axis=0) if os.path.exists(rank1_path) else None
    fake_map = np.load(fake_path).mean(axis=0)
    
    plot_enhanced_heatmap(size, r0_map, 'Rank 0 Saliency', 'rank0', 'hot')
    plot_enhanced_heatmap(size, fake_map, 'Fake Saliency', 'fake', 'hot')
    plot_enhanced_heatmap(size, r0_map - fake_map, 'Rank 0 - Fake Difference', 'diff', 'coolwarm')
    
    if r1_map is not None:
        plot_enhanced_heatmap(size, r1_map, 'Rank 1 Saliency', 'rank1', 'hot')

process_folder(100)
process_folder(200)
