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
from multiprocessing import Process, Manager

Q = np.array([
    [0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0]
])
Q_unique = np.unique(Q, axis=0)
J, K = Q_unique.shape


if __name__ == "__main__":
    rs = incomplete_global_identifiability(Q = Q_unique, uniout=True, check_level=K-1)
    
    # Print the variables
    print("Q:\n", Q)
    print("Q_unique:\n", Q_unique)
    print("J:", J)
    print("K:", K)
    print("rs:\n", rs)
    
    # Write the final rs into a CSV file
    with open('../data/fraction_subtraction_data_Q.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rs)
