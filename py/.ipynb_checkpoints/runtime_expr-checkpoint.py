#!/usr/bin/env python
import os
import sys
import time
import csv
import itertools
import numpy as np

from reduceQ import (
    get_reduced, 
    get_Qunique_from_Qreduced, 
    get_Q_from_Qunique, 
    get_Q_from_Qreduced
)
from direct_check_id import direct_check_id
from idQ import (
    Phi_mat,
    thmCheck,
    check_two_column_submatrices,
    check_three_column_submatrices,
    canonicalize,
    generate_unique_Q_bars,
    preserve_partial_order,
    generate_binary_vectors,
    distances2U,
    representative_node_set,
    brute_check_id
)

def identifiability_expr(Q):
    """
    Check if Q is identifiable, returning:
      (status, Q_bar, branch, runtime, direct_flag, brute_flag)

    status == 1 -> Q is identifiable
    status == 0 -> Q is not identifiable
    Q_bar -> witness if status==0, else []
    branch -> string describing which step concluded:
        'trivial', 'two_column', 'three_column',
        'direct_check', 'brute_candidate', 'brute_check',
        'generator_empty', 'candidate_found', 'none'

    runtime -> time in seconds

    direct_flag == 1 if direct_check_id(Q_reduced) is True, else 0
    brute_flag == 1 if Q_reduced is deemed identifiable by brute_check_id, else 0
    """
    start_time = time.time()

    # Step 1: get Q_reduced and Q_unique
    (Q_reduced, reduced_to_original, orig_indices_for_reduced,
     Q_unique, unique_to_original, reduced_to_unique) = get_reduced(Q.copy())
    
    J_reduced, K = Q_reduced.shape

    direct_flag = 0
    brute_flag = 0

    # Step 2: Check trivial columns in Q_reduced
    for k in range(K):
        if np.all(Q_reduced[:, k] == 0):
            Q_reduced_bar = Q_reduced.copy()
            Q_reduced_bar[:, k] = 1
            print("Q_reduced is trivially not identifiable (zero column).")
            Q_bar = get_Q_from_Qreduced(Q_reduced_bar, reduced_to_original)
            return 0, [Q_bar], 'trivial', time.time() - start_time, direct_flag, brute_flag
        if np.all(Q_reduced[:, k] == 1):
            Q_reduced_bar = Q_reduced.copy()
            Q_reduced_bar[:, k] = 0
            print("Q_reduced is trivially not identifiable (one column).")
            Q_bar = get_Q_from_Qreduced(Q_reduced_bar, reduced_to_original)
            return 0, [Q_bar], 'trivial', time.time() - start_time, direct_flag, brute_flag

    # Step 3: two-column check
    candidate = check_two_column_submatrices(Q_reduced)
    if candidate is not None:
        print("Q_reduced is not identifiable (two-column check).")
        Q_reduced_bar = candidate
        Q_bar = get_Q_from_Qreduced(Q_reduced_bar, reduced_to_original)
        return 0, [Q_bar], 'two_column', time.time() - start_time, direct_flag, brute_flag
    
    # Step 4: three-column check
    candidate = check_three_column_submatrices(Q_reduced)
    if candidate is not None:
        print("Q_reduced is not identifiable (three-column check).")
        Q_reduced_bar = candidate
        Q_bar = get_Q_from_Qreduced(Q_reduced_bar, reduced_to_original)
        return 0, [Q_bar], 'three_column', time.time() - start_time, direct_flag, brute_flag

    # Step 5: direct_check_id / brute_check_id
    if direct_check_id(Q_reduced):
        print("Q_reduced is identifiable (direct_check).")
        direct_flag = 1
        branch_reduced = 'direct_check'
    else:
        status_b, Q_reduced_bar = brute_check_id(Q_reduced)
        if status_b == 0:
            print("Q_reduced is not identifiable (brute_check_id).")
            Q_bar = get_Q_from_Qreduced(Q_reduced_bar, reduced_to_original)
            return 0, [Q_bar], 'brute_candidate', time.time() - start_time, direct_flag, brute_flag
        else:
            print("Q_reduced is identifiable (brute_check).")
            brute_flag = 1
            branch_reduced = 'brute_check'

    # Q_reduced is identifiable => expansions on Q_unique
    J_unique, K_uniq = Q_unique.shape
    distances = distances2U(Q_unique)
    non_generated_indices = [u for u, mapping in enumerate(reduced_to_unique) if len(mapping) == 1]
    irreplaceable_rows = set(non_generated_indices)
    replaceable_rows = set(range(J_unique)) - irreplaceable_rows
    replacement_indices = list(replaceable_rows)

    # Build all possible binary vectors (K_uniq bits) that are not in R(Q_unique)
    all_vecs = list(itertools.product([0,1], repeat=K_uniq))
    rep_set = representative_node_set(Q_unique)
    cand_bars = [np.array(vec) for vec in all_vecs if tuple(vec) not in rep_set]

    subQ_unique_bars = []
    for index in replacement_indices:
        norm_threshold = K_uniq - distances[index]
        q_bars = [cand for cand in cand_bars if np.sum(cand) <= norm_threshold]
        valid_q_bars = []
        Q_temp = Q_unique.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q_unique, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_unique_bars.append(valid_q_bars)
    
    # Cartesian product
    subQ_unique_bars = itertools.product(*subQ_unique_bars)
    Q_unique_bar_gen = generate_unique_Q_bars(subQ_unique_bars, Q_unique, replacement_indices)
    
    try:
        first_candidate = next(Q_unique_bar_gen)
    except StopIteration:
        # Means expansions gave no alternative => Q is identifiable
        print("Q is identifiable (candidate generation produced no alternative).")
        return 1, [], 'generator_empty', time.time() - start_time, direct_flag, brute_flag
    else:
        Q_unique_bar_gen = itertools.chain([first_candidate], Q_unique_bar_gen)
    
    PhiQ = Phi_mat(Q_unique)
    for Q_unique_bar in Q_unique_bar_gen:
        PhiB = Phi_mat(Q_unique_bar)
        if thmCheck(PhiQ, PhiB, tol=1e-8):
            print("Q is not identifiable (candidate found).")
            return 0, Q_unique_bar, 'candidate_found', time.time() - start_time, direct_flag, brute_flag

    print("Q is identifiable.")
    return 1, [], 'none', time.time() - start_time, direct_flag, brute_flag


def runtime_expr(J, K, N, seed, output_csv=None):
    """
    Randomly sample N binary matrices of shape (J, K). For each, call identifiability_expr(Q),
    which returns (status, Q_bar, branch, runtime, direct_flag, brute_flag).
    
    We aggregate:
      - total/average runtime
      - how often each branch occurred
      - proportion of identifiable vs. non-identifiable
      - how many times direct_flag == 1
      - how many times brute_flag == 1
    and append the results to a CSV file.

    Returns a dictionary with summarized results.
    """
    np.random.seed(seed)
    
    total_time = 0.0
    count_identifiable = 0
    count_non_identifiable = 0
    count_direct_flag = 0
    count_brute_flag = 0

    branch_counts = {}

    for i in range(N):
        Q = np.random.randint(0, 2, size=(J, K))
        # identifiability_expr(Q) now returns 6 items
        status, Q_bar, branch, runtime, direct_flag, brute_flag = identifiability_expr(Q)
        
        total_time += runtime
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        
        if status == 1:
            count_identifiable += 1
        else:
            count_non_identifiable += 1
        
        # Sum up direct/brute flags
        count_direct_flag += direct_flag
        count_brute_flag += brute_flag

    avg_runtime = total_time / N
    prop_identifiable = count_identifiable / N
    prop_non_identifiable = count_non_identifiable / N

    results = {
        'J': J,
        'K': K,
        'N': N,
        'seed': seed,
        'avg_runtime': avg_runtime,
        'prop_identifiable': prop_identifiable,
        'prop_non_identifiable': prop_non_identifiable,
        'total_time': total_time,
        'count_direct_flag': count_direct_flag,
        'prop_direct_flag': count_direct_flag / N,
        'count_brute_flag': count_brute_flag,
        'prop_brute_flag': count_brute_flag / N
    }

    # Add branch counts and proportions
    for b, cnt in branch_counts.items():
        results[f'count_{b}'] = cnt
        results[f'prop_{b}'] = cnt / N

    if output_csv is None:
        output_csv = f"data/runtime_expr_results_J{J}_K{K}.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    return results


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python runtime_expr.py <J> <K> <N> <seed>")
        sys.exit(1)

    J = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    seed = int(sys.argv[4])

    results = runtime_expr(J, K, N, seed)
    print("Simulation results:")
    print(results)
