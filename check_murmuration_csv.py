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
    # Key: rank (int), Value: {'sum': np.array, 'count': int, 'max_len': int}
    rank_stats = {}
    
    try:
        with open(INPUT_CSV, 'r') as f:
            reader = csv.reader(f)
            
            for row in tqdm(reader, desc="Processing rows"):
                if not row:
                    continue
                    
                # Skip header if present
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
                        rank = -1 # Mark as unknown/fake
                    except ValueError:
                        continue

                if aps_array is None or len(aps_array) == 0:
                    continue
                    
                current_len = len(aps_array)
                
                # Initialize rank entry if not exists
                if rank not in rank_stats:
                    rank_stats[rank] = {
                        'sum': np.zeros(current_len, dtype=np.float64),
                        'count': 0,
                        'max_len': current_len
                    }

                stats = rank_stats[rank]
                
                # Update sums with dynamic resizing
                if current_len > stats['max_len']:
                    # Expand sum array
                    new_sums = np.zeros(current_len, dtype=np.float64)
                    new_sums[:stats['max_len']] = stats['sum']
                    stats['sum'] = new_sums
                    stats['max_len'] = current_len
                elif current_len < stats['max_len']:
                    # Pad current array
                    padded = np.zeros(stats['max_len'], dtype=np.float64)
                    padded[:current_len] = aps_array
                    aps_array = padded
                
                stats['sum'] += aps_array
                stats['count'] += 1

        if not rank_stats:
            print("No valid data found.")
            return

        print(f"Found ranks: {sorted(rank_stats.keys())}")
        for r in sorted(rank_stats.keys()):
            print(f"  Rank {r}: {rank_stats[r]['count']} curves")
        
        plt.figure(figsize=(12, 6))
        
        # Get a colormap to assign different colors to different ranks
        # We use a qualitative colormap
        sorted_ranks = sorted(rank_stats.keys())
        cmap = plt.get_cmap('tab10') 
        
        max_plot_len = 0
        
        for i, r in enumerate(sorted_ranks):
            stats = rank_stats[r]
            count = stats['count']
            if count == 0:
                continue
                
            ap_avg = stats['sum'] / count
            primes = np.array([sympy.prime(k) for k in range(1, len(ap_avg) + 1)])
            
            # Labeling
            if r == -1:
                label = f'Unknown/Fake (N={count})'
                color = 'gray'
                alpha = 0.3
            else:
                label = f'Rank {r} (N={count})'
                # Cycle through colors if more than 10 ranks
                color = cmap(i % 10)
                alpha = 0.6

            plt.plot(primes, ap_avg, '.', markersize=3, alpha=alpha, color=color, label=label)
            max_plot_len = max(max_plot_len, len(ap_avg))

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