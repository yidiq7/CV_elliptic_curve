import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import sympy
from tqdm import tqdm
import os
import argparse

# Configuration
OUTPUT_PLOT = 'murmuration_plot_csv.png'

def main():
    parser = argparse.ArgumentParser(description='Check murmuration in elliptic curve data.')
    parser.add_argument('input_csv', nargs='?', default='ap.csv', help='Path to the input CSV file')
    args = parser.parse_args()
    
    INPUT_CSV = args.input_csv

    if not os.path.exists(INPUT_CSV):
        print(f"Error: '{INPUT_CSV}' not found.")
        print("Please ensure the file exists in the current directory.")
        return

    print(f"Reading data from {INPUT_CSV}...")
    
    # Dictionary to store stats per rank
    # Key: rank (int), Value: [ap_sums, count, max_len]
    rank_data = {}
    
    # Helper to update sums (Keep EXACTLY same logic as working version)
    def update_sums(sums, max_l, current_l, arr):
        if sums is None:
            return arr, current_l
        if current_l > max_l:
            new_sums = np.zeros(current_l, dtype=np.float64)
            new_sums[:max_l] = sums
            sums = new_sums
            max_l = current_l
        elif current_l < max_l:
            padded = np.zeros(max_l, dtype=np.float64)
            padded[:current_l] = arr
            arr = padded
        return sums + arr, max_l

    try:
        with open(INPUT_CSV, 'r') as f:
            reader = csv.reader(f)
            
            for row in tqdm(reader, desc="Processing rows"):
                if not row:
                    continue
                    
                if row[0].strip() and not row[0].lstrip('-').replace('.','',1).isdigit():
                    continue

                aps_array = None
                rank = -1 # Default/Unknown

                # Detect Format
                # Format 1: Real Data [conductor, rank, "[ap1, ap2, ...]"]
                if len(row) >= 3 and isinstance(row[2], str) and row[2].strip().startswith('['):
                    try:
                        rank = int(row[1])
                        aps = ast.literal_eval(row[2])
                        aps_array = np.array(aps, dtype=np.float64)
                    except (ValueError, SyntaxError):
                        continue
                
                # Format 2: Fake/Flat Data [ap1, ap2, ap3, ...]
                else:
                    try:
                        aps_array = np.array([float(x) for x in row], dtype=np.float64)
                        rank = -1 # Mark as unknown
                    except ValueError:
                        continue

                if aps_array is None or len(aps_array) == 0:
                    continue
                    
                current_len = len(aps_array)
                
                # Initialize rank entry if not exists
                if rank not in rank_data:
                    rank_data[rank] = [None, 0, 0] # [sums, count, max_len]
                
                stats = rank_data[rank]
                stats[0], stats[2] = update_sums(stats[0], stats[2], current_len, aps_array)
                stats[1] += 1

        total_count = sum(stats[1] for stats in rank_data.values())
        if total_count == 0:
            print("No valid data found.")
            return

        sorted_ranks = sorted(rank_data.keys())
        print(f"Processed {total_count} curves. Ranks found: {sorted_ranks}")
        for r in sorted_ranks:
            print(f"  Rank {r}: {rank_data[r][1]} curves")
        
        plt.figure(figsize=(12, 6))
        
        # Get a colormap for multiple ranks
        cmap = plt.get_cmap('tab10')
        
        for i, rank in enumerate(sorted_ranks):
            if rank not in [0, 1, 2]:
                continue
                
            sums, count, max_l = rank_data[rank]
            if count > 0:
                ap_avg = sums / count
                primes = np.array([sympy.prime(k) for k in range(1, len(ap_avg) + 1)])
                
                label = f'Rank {rank} (N={count})'
                color = cmap(i % 10)
                alpha = 0.5
                
                plt.plot(primes, ap_avg, '.', markersize=2, alpha=alpha, color=color, label=label)

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Prime p')
        plt.ylabel('Average a_p')
        plt.title(f'Murmuration of Elliptic Curves by Rank\nAverage $a_p$ vs $p$ from {INPUT_CSV}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_filename = f"murmuration_rank_{os.path.basename(INPUT_CSV).replace('.csv','')}.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
