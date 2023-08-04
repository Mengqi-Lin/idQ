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
             open(f'../data/incomplete_global/all_J{J}_K{K}_C{C}.csv', 'w', newline='') as allinfo_file:
            writer = csv.writer(f)
            allinfo_writer = csv.writer(allinfo_file)
            for Q in generate_canonical_matrices(J, K):
                start_time = time.time()
                
                # Create a Manager object to share variables between processes
                manager = Manager()
                return_dict = manager.dict()

                # Define a new function that calls incomplete_global_identifiability and stores the result in the shared dictionary
                def func(Q, return_dict):
                    return_dict['result'] = incomplete_global_identifiability(Q = Q, uniout=True, check_level=C)

                # Create a new Process object to run func
                p = Process(target=func, args=(Q, return_dict))

                # Start the process
                p.start()

                # Wait for 2 minutes or until the process finishes
                p.join(120)

                # If thread is still active
                if p.is_alive():
                    print("Function call exceeded 2 minute time limit.")
                    p.terminate()
                    p.join()
                    end_time = time.time()
                    runtime = end_time - start_time
                    allinfo_writer.writerow([Q.flatten(), -1, runtime])
                    continue
                
                Id,_ = return_dict['result']
                end_time = time.time()
                runtime = end_time - start_time
                allinfo_writer.writerow([Q.flatten(), Id, runtime])
                if Id & (not check_for_identity(Q)):      
                    print(Q)
                    writer.writerow(Q.flatten())

                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
                      
