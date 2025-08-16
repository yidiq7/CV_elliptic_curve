import numpy as np
from sympy import sympify, I, pi, exp
import sympy
import ast
from PIL import Image
import os
import re
import csv

primes = np.array([sympy.prime(i) for i in range(1, 101)])  # primes[0] = 2, primes[99] = 541

#with open("ap.txt", "r") as f:
#    aplist_str = f.read()
#aplist = ast.literal_eval(aplist_str)  # [[cond, rank, sha, [a_p's]], ...]
# Note: aplist[i][3] has 3600 a_p values, but we'll use first 100 per 'size'

aplist = []
with open('ap.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)

    # Skip the header row
    header = next(reader)

    for row in reader:
        conductor = int(row[0])
        rank = int(row[1])
        aps = ast.literal_eval(row[2]) 

        aplist.append([conductor, rank, aps])

with open("chifull.txt", "r") as f:
    chifull_str = f.read().strip()

chifull_str = chifull_str.replace("^", "**")

# Step 2b: Wrap complex expressions with quotes
# This pattern looks for any expression containing zeta followed by operations or exponents
complex_pattern = r'(-?\s*[\w\s\+\-\*\/\(\)]+zeta\d+(\*\*\d+)?[\w\s\+\-\*\/\(\)]*)'
chifull_str = re.sub(complex_pattern, lambda m: '"' + m.group(0).strip() + '"', chifull_str)
# --- Step 3: Parse the string into a Python list ---
chifull = ast.literal_eval(chifull_str)

# --- Step 4: Define a function to convert zeta to exp(2 pi i / N) ---
def getroot(n, exponent=1):
    """
    Compute the nth root of unity raised to the given exponent.
    """
    return np.exp(2j * np.pi / n) ** exponent

def convert_zeta_to_exp(expr):
    """
    Converts zeta expressions to exp(2 pi i / N) and applies the exponent if necessary.
    """
    # Match patterns like -zeta22, zeta22**2, or -zeta22**5
    pattern = re.compile(r'(-?)zeta(\d+)(\*\*(\d+))?')

    # Replace each match with exp(2 pi i / N)^k while preserving the sign
    def replace_match(match):
        sign = match.group(1)  # Capture the minus sign if present
        n = int(match.group(2))  # Extract the order of the root
        exponent = int(match.group(4)) if match.group(4) else 1  # Default exponent is 1

        # Build the Python expression for the nth root of unity
        root_expr = f"({sign}getroot({n}, {exponent}))"
        return root_expr

    # Replace all occurrences of zeta terms in the expression
    expr_transformed = pattern.sub(replace_match, expr)
    return expr_transformed

# --- Step 5: Evaluate the expressions ---

def evaluate_expression(expr):
    """
    Safely evaluate a math expression with numpy and getroot.
    """
    try:
        # Convert zeta expressions to Python-compatible format
        expr_transformed = convert_zeta_to_exp(expr)
        # Evaluate the resulting expression
        result = eval(expr_transformed, {"getroot": getroot, "np": np})
        return result
    except Exception as e:
        print(f"Error evaluating expression: {expr}\n{e}")
        return expr

# --- Step 6: Recursively replace and evaluate tokens in the list ---

def replace_and_evaluate(expr):
    if isinstance(expr, list):
        return [replace_and_evaluate(item) for item in expr]
    elif isinstance(expr, str):
        # Apply zeta-to-exp transformation and evaluate
        return evaluate_expression(expr)
    return expr

# --- Step 7: Apply the transformation and evaluation ---
allchi = replace_and_evaluate(chifull)


size = 100
padchi = np.array([
    [allchi[k][m % len(allchi[k])] for m in range(size)]
    for k in range(size)  
])

re_padchi = padchi.real  # 100x100
im_padchi = padchi.imag  # 100x100

os.makedirs("./coloured", exist_ok=True)

all_img_arrays = []

for i, curve in enumerate(aplist):
    ap = curve[2][:size]  # Take first 100 a_p values from 1000
    factor = np.array(ap) / (2 * np.sqrt(primes))  # Shape (100,)
    r = 0.5 - 0.5 * factor[:, None] * re_padchi  # Shape (100, 100)
    g = 0.5 * np.ones(re_padchi.shape)
    b = 0.5 - 0.5 * factor[:, None] * im_padchi  # Shape (100, 100)

    # Stack into 100x100x3 RGB array
    img_array = np.stack([r, g, b], axis=2)
    # Save before clipping and rounding
    all_img_arrays.append(img_array)
    # Clip to [0, 1] (Hasse bound |a_p| ≤ 2√p ensures reasonable values)
    img_array = np.clip(img_array, 0, 1)
    # Scale to [0, 255] and convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    #breakpoint()
    img = Image.fromarray(img_array, 'RGB')
    img.save(f"./coloured/ECcoloured{i+1}.png")
    print(f"Saved ./coloured/ECcoloured{i+1}.png")


combined_array = np.stack(all_img_arrays)
print(f"\nCombined all arrays into a single array with shape: {combined_array.shape}")

np.save("combined_twisted_arrays.npy", combined_array)
print("\nSaved the combined array to 'combined_twisted_arrays.npy'")
