import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, RESULTS_DIR
from tqdm import tqdm
from sympy import primerange
from multiprocessing import Pool, cpu_count
import csv
import time

# --- Configuration ---
NUM_SEQUENCES = 500000  # How many "fake" L-functions to generate
NUM_AP = 1000           # The first 1000 a_p coefficients for each L-function

# Pre-calculate the first 1000 primes to be shared across all processes
PRIMES = list(primerange(1, 10000))[:NUM_AP] # A safe upper bound for the 1000th prime
MAX_BOUNDS = [2 * np.sqrt(p) for p in PRIMES]

def sample_sato_tate_angle():
    """
    Samples an angle theta from the Sato-Tate distribution, whose PDF is (2/pi)*sin^2(theta).
    This is done using rejection sampling, a standard method for non-uniform distributions.
    """
    # The maximum value of the PDF (2/pi)*sin^2(theta) is 2/pi at theta = pi/2.
    max_pdf_val = 2 / np.pi
    
    while True:
        # Step 1: Propose a point uniformly in the sampling domain.
        # Propose theta uniformly from [0, pi]
        theta_proposal = np.random.uniform(0, np.pi)
        # Propose y uniformly from [0, max_pdf_value]
        y_proposal = np.random.uniform(0, max_pdf_val)
        
        # Step 2: Calculate the PDF value at the proposed theta.
        pdf_val_at_proposal = (2 / np.pi) * (np.sin(theta_proposal) ** 2)
        
        # Step 3: Accept the proposal if y is under the curve.
        if y_proposal <= pdf_val_at_proposal:
            return theta_proposal

def generate_one_fake_ap_sequence(_=None):
    """
    Generates a single list of 1000 fake a_p coefficients.
    
    The dummy argument `_` allows this function to be used with pool.map.
    """
    np.random.seed((os.getpid() + int(time.time_ns())) % (2**32))
    ap_sequence = []
    for i, p in enumerate(PRIMES):
        # 1. Sample an angle theta from the Sato-Tate distribution.
        theta_p = sample_sato_tate_angle()
        
        # 2. This gives the normalized a_p value, x_p = cos(theta_p).
        x_p = np.cos(theta_p) # This value is in [-1, 1]
        
        # 3. Scale by 2*sqrt(p) to satisfy Hasse's bound on average.
        # The result is a float.
        ap_float = x_p * MAX_BOUNDS[i]
        
        # 4. Round to the nearest integer to get the final a_p.
        # This naturally respects the bound |a_p| <= floor(2*sqrt(p)).
        ap_int = int(np.round(ap_float))
        
        ap_sequence.append(ap_int)
        
    return ap_sequence


if __name__ == "__main__":
    # Use all available CPU cores for parallel generation
    num_cores = cpu_count()
    print(f"Starting generation of {NUM_SEQUENCES} sequences using {num_cores} CPU cores...")

    # Create a pool of worker processes
    with Pool(processes=num_cores) as pool:
        # Use pool.imap to get results as they are completed, showing a progress bar
        results = list(tqdm(pool.imap(generate_one_fake_ap_sequence, range(NUM_SEQUENCES)), total=NUM_SEQUENCES))

    print(f"\nSuccessfully generated {len(results)} sequences.")
    

    with open(os.path.join(DATA_DIR, 'fake_ap.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    print("\nDataset saved to fake_ap.csv")
