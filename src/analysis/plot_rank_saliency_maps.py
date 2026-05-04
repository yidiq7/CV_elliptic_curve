"""
Generate saliency plots from pre-computed .npy files.
Produces per-rank, all-ranks-combined, and pairwise-difference plots
for the rank classifier saliency maps.

Usage: python plot_rank_saliency_maps.py
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, RESULTS_DIR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_enhanced_heatmap(map_data, title, filepath, image_size, cmap='hot'):
    std_across_twists = map_data.std(axis=1)
    mean_across_twists = map_data.mean(axis=1)
    std_across_primes = map_data.std(axis=0)
    mean_across_primes = map_data.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{title} (N={image_size})', fontsize=16, y=0.98)

    vmax = np.percentile(np.abs(map_data), 99.5)
    vmin = -vmax if cmap == 'coolwarm' else 0

    im = ax.imshow(map_data, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Primes ($p$)', fontsize=12)
    ax.set_xlabel('Twists ($\\chi$)', fontsize=12)

    divider = make_axes_locatable(ax)

    # Marginal: projection onto primes (right side)
    ax_prime = divider.append_axes("right", size="20%", pad=0.1)
    ax_prime.plot(mean_across_twists, range(image_size), color='red', label='Mean')
    ax_prime.fill_betweenx(range(image_size),
                           mean_across_twists - std_across_twists,
                           mean_across_twists + std_across_twists,
                           color='red', alpha=0.2, label='Std')
    ax_prime.invert_yaxis()
    ax_prime.set_ylim(image_size - 0.5, -0.5)
    ax_prime.margins(y=0)
    ax_prime.set_xlabel('Abs Grad', fontsize=10)
    ax_prime.set_yticks([])
    ax_prime.grid(True, alpha=0.3)
    ax_prime.legend(loc='upper right', fontsize=8)

    # Colorbar to the right of the marginal plot
    ax_cbar = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=ax_cbar)

    # Marginal: projection onto twists (top side)
    ax_twist = divider.append_axes("top", size="20%", pad=0.1)
    ax_twist.plot(range(image_size), mean_across_primes, color='blue', label='Mean')
    ax_twist.fill_between(range(image_size),
                          mean_across_primes - std_across_primes,
                          mean_across_primes + std_across_primes,
                          color='blue', alpha=0.2, label='Std')
    ax_twist.set_xlim(-0.5, image_size - 0.5)
    ax_twist.margins(x=0)
    ax_twist.set_ylabel('Abs Grad', fontsize=10)
    ax_twist.set_xticks([])
    ax_twist.grid(True, alpha=0.3)
    ax_twist.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_for_size(image_size):
    output_dir = os.path.join(RESULTS_DIR, f'saliency_maps_rank_{image_size}')
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} not found, skipping.")
        return

    NUM_CLASSES = 3
    channel_configs = [(0, 'real_channel'), (1, 'imag_channel'), (None, 'average')]

    # Load all per-rank saliency .npy files
    saliency = {}
    for rank in range(NUM_CLASSES):
        for tc in range(NUM_CLASSES):
            key = f'rank{rank}_wrt_class{tc}'
            path = os.path.join(RESULTS_DIR, f'saliency_maps_rank_{image_size}', f'{key}_saliency_avg.npy')
            if os.path.exists(path):
                saliency[key] = np.load(path)

    if not saliency:
        print(f"No saliency .npy files found in {output_dir}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Regenerating plots for N={image_size}")
    print(f"{'='*60}")

    # Compute "all combined": weighted average of rank R saliency w.r.t. class R
    # (each curve backpropagated w.r.t. its own rank's logit)
    # Without exact counts, use equal weighting as approximation
    own_class_keys = [f'rank{r}_wrt_class{r}' for r in range(NUM_CLASSES)]
    available = [saliency[k] for k in own_class_keys if k in saliency]
    if available:
        combined_sal = np.mean(available, axis=0)
        saliency['all_combined'] = combined_sal
        np.save(os.path.join(output_dir, 'all_combined_saliency_avg.npy'), combined_sal)

    # Per-rank plots
    for rank in range(NUM_CLASSES):
        key_own = f'rank{rank}_wrt_class{rank}'
        if key_own not in saliency:
            continue

        sal = saliency[key_own]
        for ch_idx, ch_name in channel_configs:
            sal_map = sal[ch_idx] if ch_idx is not None else sal.mean(axis=0)
            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] Rank {rank} Saliency (w.r.t. class {rank})'
            fname = os.path.join(output_dir, f'enhanced_marginal_rank{rank}_{ch_name}.png')
            plot_enhanced_heatmap(sal_map, title, fname, image_size, 'hot')
            print(f"  Saved {fname}")

    # All-ranks-combined plot (each curve w.r.t. its own rank)
    if 'all_combined' in saliency:
        sal = saliency['all_combined']
        for ch_idx, ch_name in channel_configs:
            sal_map = sal[ch_idx] if ch_idx is not None else sal.mean(axis=0)
            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] All Curves Combined Saliency'
            fname = os.path.join(output_dir, f'enhanced_marginal_all_combined_{ch_name}.png')
            plot_enhanced_heatmap(sal_map, title, fname, image_size, 'hot')
            print(f"  Saved {fname}")

    # Difference maps
    for r_a, r_b in [(0, 1), (0, 2), (1, 2)]:
        key_a = f'rank{r_a}_wrt_class{r_a}'
        key_b = f'rank{r_b}_wrt_class{r_b}'
        if key_a not in saliency or key_b not in saliency:
            continue

        for ch_idx, ch_name in channel_configs:
            if ch_idx is not None:
                diff_map = saliency[key_a][ch_idx] - saliency[key_b][ch_idx]
            else:
                diff_map = saliency[key_a].mean(axis=0) - saliency[key_b].mean(axis=0)

            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] Rank {r_a} - Rank {r_b} Saliency Diff'
            fname = os.path.join(output_dir, f'enhanced_marginal_diff_r{r_a}_r{r_b}_{ch_name}.png')
            plot_enhanced_heatmap(diff_map, title, fname, image_size, 'coolwarm')
            print(f"  Saved {fname}")

    print(f"Done for N={image_size}.")


if __name__ == "__main__":
    plot_for_size(100)
    plot_for_size(200)
