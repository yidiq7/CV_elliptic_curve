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
    
    ap_sums = None
    count = 0
    max_len = 0
    
    try:
        with open(INPUT_CSV, 'r') as f:
            reader = csv.reader(f)
            
            # Heuristic to detect header or format
            # We'll peek at the first valid line
            
            for row in tqdm(reader, desc="Processing rows"):
                if not row:
                    continue
                    
                # Check for header (non-digit first char)
                # But careful, fake_ap could start with negative number? 
                # "conductor" is usually positive int.
                # If row[0] is "conductor" or similar text.
                if row[0].strip() and not row[0].lstrip('-').replace('.','',1).isdigit():
                    # Likely a header
                    continue

                aps_array = None

                # Detect Format
                # Format 1: Real Data [conductor, rank, "[ap1, ap2, ...]"]
                # Check if 3rd element exists and starts with '['
                if len(row) >= 3 and isinstance(row[2], str) and row[2].strip().startswith('['):
                    try:
                        aps = ast.literal_eval(row[2])
                        aps_array = np.array(aps, dtype=np.float64)
                    except (ValueError, SyntaxError):
                        continue
                
                # Format 2: Fake/Flat Data [ap1, ap2, ap3, ...]
                # All elements should be numbers
                else:
                    try:
                        # Try converting whole row to floats
                        aps_array = np.array([float(x) for x in row], dtype=np.float64)
                    except ValueError:
                        # If conversion fails, maybe it was a mixed row or header we missed
                        continue

                if aps_array is None or len(aps_array) == 0:
                    continue
                    
                current_len = len(aps_array)
                
                if ap_sums is None:
                    ap_sums = aps_array
                    max_len = current_len
                else:
                    if current_len > max_len:
                        new_sums = np.zeros(current_len, dtype=np.float64)
                        new_sums[:max_len] = ap_sums
                        ap_sums = new_sums
                        max_len = current_len
                    elif current_len < max_len:
                        padded = np.zeros(max_len, dtype=np.float64)
                        padded[:current_len] = aps_array
                        aps_array = padded
                    
                    ap_sums += aps_array
                    
                count += 1

        if count == 0:
            print("No valid data found.")
            return

        print(f"Processed {count} curves.")
        
        # Calculate averages
        ap_averages = ap_sums / count
        
        print(f"Generating primes for x-axis (first {len(ap_averages)})...")
        # Ensure we have enough primes
        primes = np.array([sympy.prime(i) for i in range(1, len(ap_averages) + 1)])
        
        # Plotting
        print("Plotting results...")
        plt.figure(figsize=(12, 6))
        plt.plot(primes, ap_averages, '.', markersize=2, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Prime p')
        plt.ylabel('Average a_p')
        plt.title(f'Murmuration of Elliptic Curves\nAverage $a_p$ vs $p$ (N={count}) from {INPUT_CSV}')
        plt.grid(True, alpha=0.3)
        
        output_filename = f"murmuration_{os.path.basename(INPUT_CSV).replace('.csv','')}.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
