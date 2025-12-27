import numpy as np
import sys

# File paths for your data
IMAGE_SIZE = 200
REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'

def check_file(path, name):
    print(f"\nChecking {name} ({path})...")
    try:
        # Load using mmap to avoid loading entire dataset into memory
        data = np.load(path, mmap_mode='r')
        print(f"  Shape: {data.shape}")
        
        # We want the first column of the imaginary channel.
        # Data shape is (N, SIZE, SIZE, 2).
        # Channel 0 = Real, Channel 1 = Imaginary.
        # Rows = Dim 1, Cols = Dim 2.
        # First column is index 0 in Dim 2.
        # So: data[:, :, 0, 1] extracts (N, Rows) for Col 0, Channel 1.
        
        first_col_imag = data[:, :, 0, 1]
        
        # To be efficient, we can check min and max.
        min_val = np.min(first_col_imag)
        max_val = np.max(first_col_imag)
        mean_val = np.mean(first_col_imag)
        
        unique_vals = np.unique(first_col_imag)
        
        print(f"  Unique values in first column of imaginary channel: {unique_vals}")
        
        is_constant = (min_val == max_val)
        
        if is_constant:
            if np.isclose(min_val, 0.5):
                print("  -> SUCCESS: All values are exactly 0.5.")
                print("     (Note: In the generation script, 0.5 corresponds to an imaginary part of 0).")
            elif np.isclose(min_val, 0.0):
                print("  -> SUCCESS: All values are exactly 0.0.")
            else:
                print(f"  -> WARNING: Values are constant but equal to {min_val}. Check encoding.")
        else:
            print("  -> WARNING: Values are NOT constant.")
            print(f"     Min: {min_val}")
            print(f"     Max: {max_val}")
            print(f"     Mean: {mean_val}")
            
    except FileNotFoundError:
        print(f"  ERROR: File {path} not found.")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    print(f"Checking imaginary channel for first column (untwisted)...")
    check_file(REAL_DATA_PATH, "Real Data")
    check_file(FAKE_DATA_PATH, "Fake Data")
