import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_enhanced_heatmap(size):
    folder = f'saliency_maps_{size}'
    fake_path = os.path.join(DATA_DIR, folder, 'fake_saliency_avg.npy')
    
    if not os.path.exists(fake_path):
        return

    fake = np.load(fake_path)
    fake_map = fake.mean(axis=0)  # average channels

    # We want to show standard deviation across columns (twists) for each row (prime)
    std_across_twists = fake_map.std(axis=1)
    mean_across_twists = fake_map.mean(axis=1)
    
    # We want to show standard deviation across rows (primes) for each column (twist)
    std_across_primes = fake_map.std(axis=0)
    mean_across_primes = fake_map.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the 2D heatmap
    im = ax.imshow(fake_map, cmap='hot', aspect='auto')
    ax.set_title(f'High-Contrast Fake Saliency (N={size})')
    ax.set_ylabel('Primes ($p$)')
    ax.set_xlabel('Twists ($\chi$)')

    # Create dividers for marginal plots
    divider = make_axes_locatable(ax)
    
    # Marginal plot for Primes (Right side)
    ax_prime = divider.append_axes("right", size="20%", pad=0.1)
    ax_prime.plot(mean_across_twists, range(size), color='red', label='Mean')
    ax_prime.fill_betweenx(range(size), 
                           mean_across_twists - std_across_twists, 
                           mean_across_twists + std_across_twists, 
                           color='red', alpha=0.3, label='Std Dev')
    ax_prime.invert_yaxis()  # Match image coordinates
    ax_prime.set_xlabel('Saliency')
    ax_prime.set_yticks([])
    ax_prime.grid(True, alpha=0.3)
    
    # Marginal plot for Twists (Top side)
    ax_twist = divider.append_axes("top", size="20%", pad=0.1)
    ax_twist.plot(range(size), mean_across_primes, color='blue')
    ax_twist.fill_between(range(size), 
                          mean_across_primes - std_across_primes, 
                          mean_across_primes + std_across_primes, 
                          color='blue', alpha=0.3)
    ax_twist.set_ylabel('Saliency')
    ax_twist.set_xticks([])
    ax_twist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, f'enhanced_marginal_saliency_{size}.png'), dpi=300)
    plt.close()

plot_enhanced_heatmap(100)
plot_enhanced_heatmap(200)
ed_heatmap(100)
plot_enhanced_heatmap(200)
