#!/usr/bin/env python
import os
import sys
import time
import csv
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from Qbasis import (
    get_basis, 
    get_Qunique_from_Qbasis, 
    get_Q_from_Qunique, 
    get_Q_from_Qbasis
)
from solve_IP import solve_IP, solve_IP_fast
from solve_SAT import solve_SAT, solve_SAT_fast

from idQ import (
    check_two_column_submatrices,
    check_three_column_submatrices,
    contains_identity_submatrix,
    lex_sort_columns
)

def identifiability_expr(Q, solver):
    Q = Q.copy()
    
    # Step 1: get Q_basis
    (Q_basis, basis_to_original, orig_indices_for_basis,
         Q_unique, unique_to_original, basis_to_unique) = get_basis(Q)
    
    J_basis, K = Q_basis.shape
    
    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        print("Q_basis with dimensions J < K, thus not identifiable.")
        return 0, Q_bar, -1
    # All subsequent candidate generation is performed on Q_basis.
    # Step 3: Check for trivial non-identifiability on Q_basis.
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            print("Q is trivially not identifiable (all zero column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar, 0
        if np.all(Q_basis[:, k] == 1):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            print("Q is trivially not identifiable (all one column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar, 1

    candidate = check_two_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (two-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar, 2

    candidate = check_three_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (three-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar, 3
        
        
    # Step 4: Determine identifiability of Q_basis.
    if contains_identity_submatrix(Q_basis):
        print("Q is identifiable (direct_check).")
        return 1, None, 4
    else:
        
        if solver == -1:
            solution = solve_SAT_fast(Q_basis)
            if solution is not None:
                Q_basis_bar = solution
                Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
                return 0, Q_bar, 5
            else:
                return True, None, 6  # No solution exists
        elif solver == 0:
            Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)
            solution = solve_SAT(Q_sorted, solver_name='cadical195')
        elif solver == 1:
            Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)
            solution = solve_SAT(Q_sorted, solver_name='Glucose42')
        elif solver == 2:
            Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)
            solution = solve_SAT(Q_sorted, solver_name='Glucose42')            


        if solution is not None:
            Q_basis_bar = solution[:, sorted_to_original]
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar, 5
        else:
            return True, None, 6  # No solution exists
            
            
def runtime_expr(J, K, N, p, seed, solver = -1, output_csv=None):
    """
    Randomly sample N binary matrices of shape (J, K) with entries drawn from Bernoulli(p).
    For each matrix, call identifiability_expr(Q) and measure its runtime.
    
    Aggregates:
      - total/average runtime,
      - the proportion of simulations where Q is identifiable,
      - separate branch counts for branches -1, 0, 1, 2, 3, 4, 5, 6.
    
    The results are appended to a CSV file, and a dictionary with summarized results is returned.
    """
    RR = []
    np.random.seed(seed)
    for i in range(N):
        print(i)
        # Generate Q with Bernoulli(p): each entry is 1 with probability p, else 0.
        Q = np.random.binomial(1, p, size=(J, K))
        print(f"Q: {Q}")
        start_time = time.perf_counter()
        status, Q_bar, branch = identifiability_expr(Q, solver)
        if status:
            print(f"Q is identifiable")
        else:
            print(f"Q is not identifiable, and Q_bar is {Q_bar}")
        
        print(f"branch at {branch}")
        runtime = time.perf_counter() - start_time
        
        print(f"runtime for solving id of Q is {runtime}")

        results = {
            'J': J,
            'K': K,
            'N': N,
            'p': p,
            'seed': seed,
            'sim': i,
            'runtime': runtime,
            'identifiable': status,
            'branch': branch
        }

        if output_csv is None:
            output_csv = f"../data/solver{solver}_J{J}_K{K}_p{p}.csv"

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        file_exists = os.path.exists(output_csv)
        with open(output_csv, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        RR.append(results)
        print("Simulation results:")
        print(results)
    return RR

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: python runtime_expr.py <J> <K> <N> <p> <seed> <solver>")
        sys.exit(1)

    J = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    p = float(sys.argv[4])
    seed = int(sys.argv[5])
    solver = int(sys.argv[6])
    
    RR = runtime_expr(J, K, N, p, seed, solver)

    print("Simulation results:")
    print(RR)
