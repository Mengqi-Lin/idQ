#!/usr/bin/env python
import os
import sys
import time
import csv
import itertools
import numpy as np
import networkx as nx
from idQ import (
    Phi_mat,
    thmCheck,
    check_two_column_submatrices,
    fix_submatrix,
    canonicalize,
    generate_unique_Q_bars,
    preserve_partial_order,
    generate_binary_vectors,
    distances2U
)

def identifiability(Q):
    """
    Check identifiability of Q.
    Returns a tuple: (status, Q_bar, branch) where status==1 means Q is identifiable,
    status==0 means not identifiable, Q_bar is a candidate matrix showing non-identifiability (if any),
    and branch indicates which step triggered the return:
      - 'trivial': Q contains an all-zero or all-one column.
      - 'two_column': determined via the two-column submatrix check.
      - 'generator_empty': candidate generation produced no alternative (Q is identifiable).
      - 'candidate_found': a candidate Q_bar was found that satisfies the algebraic condition.
    """
    Q = Q.copy()
    # Check if Q is a binary matrix.
    if not isinstance(Q, np.ndarray) or not np.all((Q == 0) | (Q == 1)):
        raise Exception("Q must be a binary matrix.")
    
    # Remove all-zero rows.
    row_sums = np.sum(Q, axis=1)
    if np.any(row_sums == 0):
        Q = Q[row_sums != 0]
        print("All-zero rows have been removed from Q.")
    
    # Remove identical rows.
    Q_unique = np.unique(Q, axis=0)
    if len(Q_unique) != len(Q):
        Q = Q_unique
        print("Removed identical rows of Q.")
    
    # Check for trivial non-identifiability: all-zero or all-one columns.
    for k in range(Q.shape[1]):
        if np.all(Q[:, k] == 0):
            Q_bar = Q.copy()
            Q_bar[:, k] = 1  # Replace the k-th column with all ones.
            print("Q is trivially not identifiable (zero column).")
            return 0, [Q_bar], 'trivial'
        if np.all(Q[:, k] == 1):
            Q_bar = Q.copy()
            Q_bar[:, k] = 0  # Replace the k-th column with all zeros.
            print("Q is trivially not identifiable (one column).")
            return 0, [Q_bar], 'trivial'
    
    # Step 2: Two-column submatrix check.
    Q_bar_candidate = check_two_column_submatrices(Q)
    if Q_bar_candidate is not None:
        print("Identifiability determined at two-column check.")
        return 0, [Q_bar_candidate], 'two_column'
    
    J, K = Q.shape
    distances = distances2U(Q)
    irreplaceable_rows = np.where(np.array(distances) == K - 1)[0]
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    replacement_indices = list(replaceable_rows)
    subQ_bars = []
    for i in range(len(replaceable_rows)):
        index = replacement_indices[i]
        q_bars = []
        for p in range(1, K - distances[index] + 1):
            q_bars.extend(generate_binary_vectors(K, p))
        valid_q_bars = []
        Q_temp = Q.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q, replacement_indices)
    
    try:
        first_Q_bar = next(Q_bar_gen)
    except StopIteration:
        print("Q is identifiable (candidate generation produced no alternative).")
        return 1, [], 'generator_empty'
    else:
        Q_bar_gen = itertools.chain([first_Q_bar], Q_bar_gen)
    
    PhiQ = Phi_mat(Q)
    for Q_bar in Q_bar_gen:
        # Skip candidates that are permutation equivalent to Q.
        if np.array_equal(canonicalize(Q_bar), canonicalize(Q)):
            continue
        PhiB = Phi_mat(Q_bar)
        if thmCheck(PhiQ, PhiB, tol=1e-8):
            print("Q is not identifiable (candidate found).")
            return 0, Q_bar, 'candidate_found'
    
    print("Q is identifiable.")
    return 1, [], 'none'

def runtime_expr(J, K, N, output_csv='data/runtime_expr_results.csv'):
    """
    Randomly sample N binary matrices of shape (J, K), check identifiability,
    and compute:
      1) Average runtime for identifiability(Q)
      2) Proportion of matrices determined non-identifiable by trivial or two-column check.
      3) Proportion of matrices for which candidate generation produced no alternative.
      4) Overall proportion of identifiable vs. non-identifiable matrices.
    
    The results are appended to a CSV file located in a folder called 'logs'.
    
    Returns a dictionary with the results.
    """
    total_time = 0.0
    count_trivial = 0
    count_two_column = 0
    count_generator_empty = 0
    count_candidate_found = 0
    count_identifiable = 0
    count_non_identifiable = 0

    for i in range(N):
        Q = np.random.randint(0, 2, size=(J, K))
        start = time.time()
        status, Q_bar, branch = identifiability(Q)
        runtime = time.time() - start
        total_time += runtime
        
        if branch == 'trivial':
            count_trivial += 1
        elif branch == 'two_column':
            count_two_column += 1
        elif branch == 'generator_empty':
            count_generator_empty += 1
        elif branch == 'candidate_found':
            count_candidate_found += 1
        
        if status == 1:
            count_identifiable += 1
        else:
            count_non_identifiable += 1

    avg_runtime = total_time / N
    prop_no_candidate = (count_trivial + count_two_column) / N
    prop_generator_empty = count_generator_empty / N
    prop_identifiable = count_identifiable / N
    prop_non_identifiable = count_non_identifiable / N
    
    results = {
        'J': J,
        'K': K,
        'N': N,
        'avg_runtime': avg_runtime,
        'prop_no_candidate': prop_no_candidate,  # trivial or two_column branch
        'prop_generator_empty': prop_generator_empty,
        'prop_identifiable': prop_identifiable,
        'prop_non_identifiable': prop_non_identifiable,
        'count_trivial': count_trivial,
        'count_two_column': count_two_column,
        'count_generator_empty': count_generator_empty,
        'count_candidate_found': count_candidate_found
    }
    
    # Ensure the logs directory exists.
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Append results to CSV file.
    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    return results

if __name__ == '__main__':
    # Command-line arguments: J, K, N.
    if len(sys.argv) != 4:
        print("Usage: python runtime_expr.py <J> <K> <N>")
        sys.exit(1)
    J = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    
    results = runtime_expr(J, K, N)
    print("Simulation results:")
    print(results)
