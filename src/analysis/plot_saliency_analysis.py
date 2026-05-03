import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import os

def analyze_saliency(folder, size):
    rank0_path = os.path.join(RESULTS_DIR, folder, 'rank0_saliency_avg.npy')
    rank1_path = os.path.join(RESULTS_DIR, folder, 'rank1_saliency_avg.npy')
    fake_path = os.path.join(RESULTS_DIR, folder, 'fake_saliency_avg.npy')
    
    if not (os.path.exists(rank0_path) and os.path.exists(rank1_path) and os.path.exists(fake_path)):
        print(f"Skipping {folder}, missing data files.")
        return

    rank0 = np.load(rank0_path)
    rank1 = np.load(rank1_path)
    fake = np.load(fake_path)

    # Average across channels (Real/Imag parts)
    r0_map = rank0.mean(axis=0)
    r1_map = rank1.mean(axis=0)
    fake_map = fake.mean(axis=0)

    # Axis 0 = Primes, Axis 1 = Twists (from generator script)

    # 1. 1D Projection over Primes (Averaging over twists, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(r0_map.mean(axis=1), label='Rank 0', color='blue', alpha=0.8)
    plt.plot(r1_map.mean(axis=1), label='Rank 1', color='red', alpha=0.8)
    plt.plot(fake_map.mean(axis=1), label='Fake', color='black', linestyle='--', alpha=0.8)
    
    plt.title(f'Saliency vs Prime Index (N={size})')
    plt.xlabel('Prime Index (p)')
    plt.ylabel('Mean Absolute Saliency (Averaged over Twists)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, f'analysis_primes_{size}.png'), dpi=300)
    plt.close()

    # 2. DIFFERENCE in Saliency across primes (axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(r0_map.mean(axis=1) - fake_map.mean(axis=1), label='Rank 0 - Fake', color='blue')
    plt.plot(r1_map.mean(axis=1) - fake_map.mean(axis=1), label='Rank 1 - Fake', color='red')
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.title(f'Differential Saliency vs Prime Index (N={size})')
    plt.xlabel('Prime Index (p)')
    plt.ylabel('Saliency Difference (Real - Fake)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, f'analysis_primes_diff_{size}.png'), dpi=300)
    plt.close()

    # 3. 1D Projection over Twists (Averaging over primes, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(r0_map.mean(axis=0), label='Rank 0', color='blue', alpha=0.8)
    plt.plot(r1_map.mean(axis=0), label='Rank 1', color='red', alpha=0.8)
    plt.plot(fake_map.mean(axis=0), label='Fake', color='black', linestyle='--', alpha=0.8)
    plt.title(f'Saliency vs Dirichlet Character Index (N={size})')
    plt.xlabel('Dirichlet Character Index ($\chi$)')
    plt.ylabel('Mean Absolute Saliency (Averaged over Primes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, f'analysis_twists_{size}.png'), dpi=300)
    plt.close()

    # 4. Identify the top 5 most important primes (not twists!)
    r0_prime_saliency = r0_map.mean(axis=1)
    top_indices = np.argsort(r0_prime_saliency)[-5:][::-1]
    print(f"\nResults for N={size}:")
    print(f"Top 5 most salient prime indices for Rank 0: {top_indices}")
    print(f"Saliency values: {r0_prime_saliency[top_indices]}")

analyze_saliency('saliency_maps_100', 100)
analyze_saliency('saliency_maps_200', 200)

