# -*- coding: utf-8 -*-
"""Ex7.5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cg2X2V_B3vPT6oRI2sA1GRH7NVhXzPKb
"""

import random
from mpmath import mp

def compute_k_shingles(s: str, k: int):
    """
    Compute k-shingles for a given string of digits.
    Return a sorted list of integers corresponding to the k-shingles.
    """
    shingles = set()
    for i in range(len(s) - k + 1):
        k_shingle_str = s[i:i+k]
        k_shingle_int = int(k_shingle_str)
        shingles.add(k_shingle_int)
    return sorted(shingles)

def generate_hash_functions(K, N):
    """
    Generate K hash functions of the form h(x) = ((a*x + b) % p) % N + 1.

    """
    hash_funcs = []
    p = 2**31 - 1  # Large prime
    for _ in range(K):
        a = random.randint(1, N)  # Coefficients must be > 0
        b = random.randint(0, N)
        hash_funcs.append((a, b, p))
    return hash_funcs

def parse_data(input_data):
    """
    Parse the input data structure and return it in the standardized format.
    """
    # Example: Assume input_data is a dictionary {column_id: [indices]}
    data_matrix = [set(indices) for _, indices in input_data.items()]
    return data_matrix

def generate_dataset(m, q, delta, N):
    """
    Generate a dataset with m columns, each represented as a set of unique indices.

    """
    dataset = []

    C0 = set(random.sample(range(N), q))
    dataset.append(C0)

    # Generate subsequent columns by modifying a fraction of the previous column
    for _ in range(1, m):
        prev_column = dataset[-1]
        num_replace = int(delta * q)

        # Randomly select elements to replace
        replace_indices = set(random.sample(prev_column, num_replace))

        # Generate new elements that are not in the current column
        new_indices = set()
        while len(new_indices) < num_replace:
            candidate = random.randint(0, N - 1)
            if candidate not in prev_column:
                new_indices.add(candidate)

        # Create the new column
        new_column = (prev_column - replace_indices) | new_indices
        dataset.append(new_column)

    return dataset

def minhash_signatures(data_matrix, hash_funcs, N):
    """
    Compute the MinHash signature matrix for multiple columns using given hash functions.

    """
    K = len(hash_funcs)
    m = len(data_matrix)
    signature_matrix = [[float('inf')] * m for _ in range(K)]

    for col_idx, positions in enumerate(data_matrix):
        for row_idx, (a, b, p) in enumerate(hash_funcs):
            for x in positions:
                h = ((a * x + b) % p) % N + 1
                if h < signature_matrix[row_idx][col_idx]:
                    signature_matrix[row_idx][col_idx] = h

    return signature_matrix

def compute_jaccard_similarity(dataset):
    """
    Compute the Jaccard similarity for each pair of columns in the dataset.
    """
    m = len(dataset)
    jaccard_matrix = [[0] * m for _ in range(m)]

    for i in range(m):
       for j in range(i, m):  # Use symmetry to reduce computations
            set_i = dataset[i]
            set_j = dataset[j]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            similarity = intersection / union if union > 0 else 0
            jaccard_matrix[i][j] = similarity
            jaccard_matrix[j][i] = similarity  # Symmetric value

    return jaccard_matrix

# Main
if __name__ == "__main__":

    m = 100        # Number of columns
    q = 20000      # Number of non-zero positions per column
    delta = 0.02   # Fraction of elements to replace
    N = 10**8      # Range of possible set elements

    # Generate dataset
    dataset = generate_dataset(m, q, delta, N)

    K = 100
    hash_funcs = generate_hash_functions(K, N)

    signature_matrix = minhash_signatures(dataset, hash_funcs, N)

    # Output first 5 rows of the signature matrix for inspection
    print("MinHash signature matrix (first 5 rows):")
    for row_idx, row in enumerate(signature_matrix[:5]):
        print(f"Hash function {row_idx + 1}: {row}")

    jaccard_matrix = compute_jaccard_similarity(dataset)
    print("\nJaccard similarity matrix (first 5x5 submatrix):")
    for row in jaccard_matrix[:5]:
        print(row[:5])