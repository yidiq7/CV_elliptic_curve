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
    
    # Separate storage for Rank 0 and Rank 1
    ap_sums_r0 = None
    count_r0 = 0
    max_len_r0 = 0
    
    ap_sums_r1 = None
    count_r1 = 0
    max_len_r1 = 0
    
    # Fallback for data without rank (e.g. fake data)
    ap_sums_unknown = None
    count_unknown = 0
    max_len_unknown = 0
    
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
                
                # Helper to update sums
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

                if rank == 0:
                    ap_sums_r0, max_len_r0 = update_sums(ap_sums_r0, max_len_r0, current_len, aps_array)
                    count_r0 += 1
                elif rank == 1:
                    ap_sums_r1, max_len_r1 = update_sums(ap_sums_r1, max_len_r1, current_len, aps_array)
                    count_r1 += 1
                else:
                    ap_sums_unknown, max_len_unknown = update_sums(ap_sums_unknown, max_len_unknown, current_len, aps_array)
                    count_unknown += 1

        total_count = count_r0 + count_r1 + count_unknown
        if total_count == 0:
            print("No valid data found.")
            return

        print(f"Processed {total_count} curves (Rank 0: {count_r0}, Rank 1: {count_r1}, Unknown/Other: {count_unknown}).")
        
        plt.figure(figsize=(12, 6))
        
        max_plot_len = 0
        
        # Plot Rank 0
        if count_r0 > 0:
            ap_avg_r0 = ap_sums_r0 / count_r0
            primes_r0 = np.array([sympy.prime(i) for i in range(1, len(ap_avg_r0) + 1)])
            plt.plot(primes_r0, ap_avg_r0, '.', markersize=2, alpha=0.5, color='blue', label=f'Rank 0 (N={count_r0})')
            max_plot_len = max(max_plot_len, len(ap_avg_r0))
            
        # Plot Rank 1
        if count_r1 > 0:
            ap_avg_r1 = ap_sums_r1 / count_r1
            primes_r1 = np.array([sympy.prime(i) for i in range(1, len(ap_avg_r1) + 1)])
            plt.plot(primes_r1, ap_avg_r1, '.', markersize=2, alpha=0.5, color='orange', label=f'Rank 1 (N={count_r1})')
            max_plot_len = max(max_plot_len, len(ap_avg_r1))

        # Plot Unknown (e.g. Fake data)
        if count_unknown > 0:
            ap_avg_unk = ap_sums_unknown / count_unknown
            primes_unk = np.array([sympy.prime(i) for i in range(1, len(ap_avg_unk) + 1)])
            plt.plot(primes_unk, ap_avg_unk, '.', markersize=2, alpha=0.3, color='gray', label=f'Unknown Rank (N={count_unknown})')
            max_plot_len = max(max_plot_len, len(ap_avg_unk))

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
