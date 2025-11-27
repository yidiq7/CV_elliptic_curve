import numpy as np
from sympy import sympify, I, pi, exp
import sympy
import ast
from PIL import Image
import os
import re
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

SIZE = 300
GENERATE_REAL = True 
GENERATE_FAKE = True

primes = np.array([sympy.prime(i) for i in range(1, SIZE+1)])  # primes[0] = 2, primes[99] = 541

with open("chifull.txt", "r") as f:
    chifull_str = f.read().strip()

chifull_str = chifull_str.replace("^", "**")

# Step 2b: Wrap complex expressions with quotes
complex_pattern = r'(-?\s*[\w\s\+\-\*\/\(\)]+zeta\d+(\*\*\d+)?[\w\s\+\-\*\/\(\)]*)'
chifull_str = re.sub(complex_pattern, lambda m: '"' + m.group(0).strip() + '"', chifull_str)
chifull = ast.literal_eval(chifull_str)
print('length of chifull', len(chifull))

def getroot(n, exponent=1):
    """
    Compute the nth root of unity raised to the given exponent.
    """
    return np.exp(2j * np.pi / n) ** exponent

def convert_zeta_to_exp(expr):
    """
    Converts zeta expressions to exp(2 pi i / N) and applies the exponent if necessary.
    """
    pattern = re.compile(r'(-?)zeta(\d+)(\*\*(\d+))?')

    def replace_match(match):
        sign = match.group(1)
        n = int(match.group(2))
        exponent = int(match.group(4)) if match.group(4) else 1
        root_expr = f"({sign}getroot({n}, {exponent}))"
        return root_expr

    expr_transformed = pattern.sub(replace_match, expr)
    return expr_transformed

def evaluate_expression(expr):
    """
    Safely evaluate a math expression with numpy and getroot.
    """
    try:
        expr_transformed = convert_zeta_to_exp(expr)
        result = eval(expr_transformed, {"getroot": getroot, "np": np})
        return result
    except Exception as e:
        print(f"Error evaluating expression: {expr}\n{e}")
        return expr

def replace_and_evaluate(expr):
    if isinstance(expr, list):
        return [replace_and_evaluate(item) for item in expr]
    elif isinstance(expr, str):
        return evaluate_expression(expr)
    return expr

allchi = replace_and_evaluate(chifull)

padchi = np.array([
    [allchi[k][m % len(allchi[k])] for m in range(SIZE)]
    for k in range(SIZE)  
])

re_padchi = padchi.real
im_padchi = padchi.imag

# Precompute denominator to avoid redundant calculations
sqrt_primes = 2 * np.sqrt(primes)

def twisted_image_from_ap(ap, rgb=False):
    factor = np.array(ap[:SIZE]) / sqrt_primes
    r = 0.5 - 0.5 * factor[:, None] * re_padchi
    g = 0.5 * np.ones(re_padchi.shape)
    b = 0.5 - 0.5 * factor[:, None] * im_padchi
    if rgb:
        img_array = np.stack([r, g, b], axis=2)
    else:
    img_array = np.stack([r, b], axis=2)
    return img_array

os.makedirs("./coloured", exist_ok=True)

if GENERATE_REAL:
    aplist = []
    with open('ap.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            conductor = int(row[0])
            rank = int(row[1])
            aps = ast.literal_eval(row[2]) 
            aplist.append([conductor, rank, aps])

    # Write directly to .npy file using memory-mapped array
    num_curves = len(aplist)
    combined_array = np.lib.format.open_memmap(
        f"combined_twisted_arrays_{SIZE}.npy",
        mode='w+',
        dtype=np.float64,
        shape=(num_curves, SIZE, SIZE, 3)
    )

    for i, curve in enumerate(tqdm(aplist, desc="Processing real curves", unit="curve")):
        ap = curve[2][:SIZE]
        img_array = twisted_image_from_ap(ap)
        combined_array[i] = img_array
        
        # Optional: save individual images
        # img_array_clipped = np.clip(img_array, 0, 1)
        # img_array_uint8 = (img_array_clipped * 255).astype(np.uint8)
        # img = Image.fromarray(img_array_uint8, 'RGB')
        # img.save(f"./coloured/ECcoloured{i+1}.png")

    # Flush to disk
    del combined_array
    print(f"\nSaved {num_curves} real curves to 'combined_twisted_arrays.npy'")


if GENERATE_FAKE:
    # Count rows first to pre-allocate memory-mapped array
    with open('fake_ap.csv', 'r', newline='') as f:
        num_rows = sum(1 for _ in f)
    
    print(f"Loaded {num_rows} rows from fake_ap.csv")

    # Create memory-mapped array
    combined_array_fake = np.lib.format.open_memmap(
        f"combined_twisted_arrays_fake_{SIZE}.npy",
        mode='w+',
        dtype=np.float64,
        shape=(num_rows, SIZE, SIZE, 2)
    )

    # Process in batches to control memory usage
    BATCH_SIZE = 1000
    
    with open('fake_ap.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        batch = []
        batch_start_idx = 0
        
        for i, row in enumerate(tqdm(reader, total=num_rows, desc="Processing fake curves", unit="row")):
            integer_row = [int(value) for value in row]
            batch.append(integer_row)
            
            # Process batch when full or at end of file
            if len(batch) >= BATCH_SIZE or i == num_rows - 1:
                # Process batch in parallel
                num_processes = min(cpu_count(), len(batch))
                chunk_size = max(1, len(batch) // (num_processes * 4))
                
                with Pool(processes=num_processes) as pool:
                    batch_results = list(pool.imap(
                        twisted_image_from_ap, 
                        batch, 
                        chunksize=chunk_size
                    ))
                
                # Write batch to memory-mapped array
                batch_end_idx = batch_start_idx + len(batch)
                combined_array_fake[batch_start_idx:batch_end_idx] = np.array(batch_results)
                
                # Clear batch
                batch = []
                batch_start_idx = batch_end_idx

    # Flush to disk
    del combined_array_fake
    print(f"\nSaved {num_rows} fake curves to 'combined_twisted_arrays_fake.npy'")
