import numpy as np
from itertools import combinations, product, permutations
import warnings
import time
from idQ import is_parallel, is_strictly_less_than, is_less_equal_than, generate_binary_vectors, generate_permutations, get_D_l, height_of_Q, distances2U, preserve_partial_order, local_identifiability, global_identifiability, incomplete_global_identifiability
from idQ import generate_DAG, generate_hasse_diagram, check_for_identity, topo_order
import random
from expr_function import random_generate_Q, test_local_identifiability, generate_canonical_matrices, binary_matrix_to_string, sort_lexicographically, random_generate_Q, prop_check, sort_binary_matrix
import sys
import itertools
import csv


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: script.py J K C")
        sys.exit(1)

    try:
        J = int(sys.argv[1])
        K = int(sys.argv[2])
        C = int(sys.argv[3])
    except ValueError:
        print("J, K, and C must be integers.")
        sys.exit(1)

    print(f"J: {J}")
    print(f"K: {K}")
    print(f"check_level: {C}")

    try:
        with open(f'../data/enumerate_global_idQ_J{J}_K{K}_C{C}.csv', 'w', newline='') as f, \
             open(f'../data/incomplete_global_runtime_J{J}_K{K}_C{C}.csv', 'w', newline='') as runtime_file:
            writer = csv.writer(f)
            runtime_writer = csv.writer(runtime_file)
            for Q in generate_canonical_matrices(J, K):
                start_time = time.time()
                Id,_ = incomplete_global_identifiability(Q = Q, uniout=True, check_level=C)
                end_time = time.time()
                runtime = end_time - start_time
                runtime_writer.writerow([Id, runtime])
                if Id & (not check_for_identity(Q)):      
                    print(Q)
                    writer.writerow(Q.flatten())
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
                      