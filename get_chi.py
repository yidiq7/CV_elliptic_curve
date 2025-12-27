from sage.all import DirichletGroup

# Set the maximum modulus
N_max = 500
output_filename = 'chifull_500.txt'

# This will store all the character lists
all_characters = []

print(f"Generating all Dirichlet characters for moduli 1 to {N_max}...")

# Loop through each modulus N from 1 to N_max
for N in range(1, N_max + 1):
    # Get the group of Dirichlet characters modulo N
    G = DirichletGroup(N)
    
    # Iterate through each character in the group
    for chi in G:
        # .list() returns the values [chi(0), chi(1), ..., chi(N-1)]
        all_characters.append(chi.list())

print(f"Total number of characters generated: {len(all_characters)}")

# --- Save to file ---
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Convert the entire list to its string representation
        f.write(str(all_characters))
    
    print(f"Successfully saved all characters to {output_filename}")

except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
