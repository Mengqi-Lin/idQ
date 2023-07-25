import numpy as np
import itertools
from itertools import combinations
from itertools import combinations, product
from itertools import permutations
import warnings
import time
from idQ import local_identifiability
from idQ import is_parallel
from idQ import is_strictly_less_than
from idQ import is_less_equal_than
from idQ import generate_binary_vectors
from idQ import generate_permutations 
from idQ import get_D_l
from idQ import height_of_Q
from idQ import distances2U
from idQ import preserve_partial_order
import random

def generate_Q(J, K):
    # Generate all possible binary sequences of length K
    all_sequences = list(itertools.product([0, 1], repeat=K))

    # Remove the sequence of all zeros if it's present
    all_sequences = [seq for seq in all_sequences if any(seq)]

    # Randomly select J unique sequences
    selected_sequences = random.sample(all_sequences, J)

    # Convert the selected sequences into a NumPy array
    Q = np.array(selected_sequences)
    
    # Sort rows of Q
    Q = Q[np.lexsort(Q.T[::-1])]

    return Q


def test_local_identifiability(J, K, kappas, check_levels):
    print(f"J: {J}, K: {K}")

    # Generate Q
    Q = generate_Q(J, K)
    print("Q:")
    print(Q)

    # Iterate over all combinations of kappa and check_level
    for kappa in kappas:
        for check_level in check_levels:
            print(f"kappa: {kappa}, check_level: {check_level}")
            
            # Start the timer
            start_time = time.time()
            
            # Apply local_identifiability
            result = local_identifiability(Q, kappa, uniout=False, check_level=check_level)
            print(result)
            
            # Stop the timer and calculate the elapsed time
            elapsed_time = time.time() - start_time
            
            # Print the runtime
            print(f"Runtime: {elapsed_time} seconds")
            print("---------------------------")
