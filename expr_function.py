import numpy as np
import itertools
from itertools import combinations, product, permutations
import warnings
import time
from idQ import is_parallel, is_strictly_less_than, is_less_equal_than, generate_binary_vectors, generate_permutations, get_D_l, height_of_Q, distances2U, preserve_partial_order, local_identifiability, global_identifiability, incomplete_global_identifiability
from idQ import generate_DAG, generate_hasse_diagram, check_for_identity, topo_order
import random

## Randomly generate J times K binary matrix.
def random_generate_Q(J, K):
    # Initialize an empty set to store unique sequences
    sequences = set()

    # Loop until J unique sequences are found
    while len(sequences) < J:
        # Generate a random binary sequence of length K
        seq = tuple(np.random.choice([0, 1], size=K))
        
        # If the sequence is not all zeros, add it to the set
        if any(seq):
            sequences.add(seq)
    
    # Convert the selected sequences into a NumPy array
    Q = np.array(list(sequences))

    # Sort rows of Q
    Q = Q[np.lexsort(Q.T[::-1])]

    return Q

def binary_matrix_to_string(binary_matrix):
    return '_'.join(''.join(map(str, row)) for row in binary_matrix)

def sort_lexicographically(matrix, axis):
    if axis == 0:  # sorting columns
        # Transpose matrix for column-wise operations
        matrix = matrix.T

    # Convert binary rows to integers for sorting
    int_rows = [int(''.join(map(str, row)), 2) for row in matrix]

    # Get the indices that would sort the rows
    sorted_indices = np.argsort(int_rows)

    # Sort the matrix using the sorted indices
    sorted_matrix = matrix[sorted_indices]

    if axis == 0:  # sorting columns
        # Transpose matrix back to original shape
        sorted_matrix = sorted_matrix.T

    return sorted_matrix

def generate_canonical_matrices(J, K):
    # Generate all possible binary sequences of length K (excluding all zeros)
    all_sequences = [seq for seq in itertools.product([0, 1], repeat=K) if any(seq)]
    
    seen = set()  # This will hold the strings representing the matrices we've seen
    column_canonical_matrices = []  # This will hold unique matrics under column permutation
    all_canonical_matrices = [] # This will hold the actual matrices
    
    # Combinations of J unique sequences
    for selected_sequences in itertools.combinations(all_sequences, J):
        Q = np.array(selected_sequences)
        # Canonical form of Q: sorted by columns
        Q_canonical = sort_lexicographically(Q, 0)
        # Convert to string
        Q_string = binary_matrix_to_string(Q_canonical)
        # If we haven't seen this canonical form before, add it to the 'seen' set and to the list of all matrices
        if Q_string not in seen:
            seen.add(Q_string)
            column_canonical_matrices.append(Q_canonical)
    
    seen2 = set()
    for Q in column_canonical_matrices:
        
        # Canonical form of Q: sorted by rows
        Q_canonical = sort_lexicographically(Q, 1)

        # Convert to string
        Q_string = binary_matrix_to_string(Q_canonical)
        
        # If we haven't seen this canonical form before, add it to the 'seen' set and to the list of all matrices
        if Q_string not in seen2:
            seen2.add(Q_string)
            all_canonical_matrices.append(Q_canonical)
            
    return all_canonical_matrices


def prop_check(K, check_level=3):
    I_K = np.array(generate_binary_vectors(K, 1))
    Q = I_K.copy()
    Q[K-1, :] = I_K[K-1, :] + I_K[K-2, :]
    Q = np.concatenate((Q, I_K[K-1, :][None, :] + I_K[K-3, :][None, :]), axis=0)
    return global_identifiability(Q = Q, uniout=True, check_level=check_level)


def test_local_identifiability(J, K, kappas, check_levels, seed = 0):
    print(f"J: {J}, K: {K}")
    
    # Set the seed
    print(f"seed:{seed}")
    np.random.seed(seed)
    # Generate Q
    Q = generate_Q(J, K)
    print("Q:")
    print(Q)

    # Initialize lists to store results
    identifiable_statuses = []
    elapsed_times = []

    # Iterate over all combinations of kappa and check_level
    for kappa in kappas:
        for check_level in check_levels:
            print(f"kappa: {kappa}, check_level: {check_level}")
            
            # Start the timer
            start_time = time.time()
            
            # Apply local_identifiability
            message, Q_bars = local_identifiability(Q, kappa, uniout=True, check_level=check_level)
            
            # Stop the timer and calculate the elapsed time
            elapsed_time = time.time() - start_time

            # If Q_bars is not empty, Q is not identifiable
            if Q_bars:
                identifiable_statuses.append(False)
            else:
                identifiable_statuses.append(True)
            
            # Store the elapsed time
            elapsed_times.append(elapsed_time)
            
            # Print the results
            print(message)
            print(Q_bars)
            print(f"Runtime: {elapsed_time} seconds")
            print("---------------------------")
            
    return Q, identifiable_statuses, elapsed_times
