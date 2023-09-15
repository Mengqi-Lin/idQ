import numpy as np
from itertools import combinations, product, permutations
import warnings
import time
from idQ import is_parallel, is_strictly_less_than, is_less_equal_than, generate_binary_vectors, generate_permutations, get_D_l, height_of_Q, distances2U, preserve_partial_order, local_identifiability, global_identifiability
from idQ import generate_DAG, generate_hasse_diagram, check_for_identity, topo_order
import random
from expr_function import random_generate_Q, test_local_identifiability, generate_canonical_matrices, binary_matrix_to_string, sort_lexicographically, random_generate_Q, prop_check, sort_binary_matrix
import sys
import itertools

if __name__ == "__main__":
    J = int(sys.argv[1])
    K = int(sys.argv[2])
    seed = int(sys.argv[3])
    random.seed(seed)  # Set the seed

    # Define kappas and check_levels
    kmax = np.floor(J/3)
    kappas = range(1, int(kmax)+1, 1)
    check_levels = range(3, min(J, K)+1, 1)
    
    # Print kappas and check_levels
    print(f"kappas: {list(kappas)}")
    print(f"check_levels: {list(check_levels)}")

    try:
        Q, identifiable_statuses, elapsed_times = test_local_identifiability(J, K, list(kappas), list(check_levels), seed)
        
        # Open the file in append mode. If file does not exist, it will be created.
        with open(f'exprQ_J{J}_K{K}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Write the seed, J, K, and result into the file as a new row.
            writer.writerow([J, K, seed, Q, identifiable_statuses, elapsed_times])
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
