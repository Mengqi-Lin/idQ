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


BRANCH_LABELS = {
    -1: "basis_J_less_than_K",
     0: "all_zero_column",
     1: "all_one_column",
     2: "two_column_check_failed",
     3: "three_column_check_failed",
     4: "identity_submatrix_direct_id",
     5: "SAT_found_counterexample",
     6: "SAT_unsat_identifiable",
}

### Helper functions

def has_any_pure_node(Q):
    """
    A pure node is a row equal to e_k for some k.
    Since Q is binary, this is equivalent to row sum equal to 1.
    """
    Q = np.asarray(Q)
    return bool(np.any(Q.sum(axis=1) == 1))


def violates_two_column_necessary(Q):
    """
    Checks whether some pair of columns fails to contain both (1,0) and (0,1).
    This is the pairwise necessary condition discussed in the response.
    """
    Q = np.asarray(Q)
    J, K = Q.shape

    if K < 2:
        return False

    for k1, k2 in itertools.combinations(range(K), 2):
        c1 = Q[:, k1]
        c2 = Q[:, k2]

        has_10 = np.any((c1 == 1) & (c2 == 0))
        has_01 = np.any((c1 == 0) & (c2 == 1))

        if not (has_10 and has_01):
            return True

    return False


def count_representative_classes(Q_basis, op="conj"):
    """
    Computes M = |R(Q_basis)| as the number of distinct columns of Phi(Q_basis).

    op="conj" corresponds to the conjunctive/DINA Boolean operator:
        Phi_{j,alpha}=1{q_j <= alpha}.

    op="disj" corresponds to the disjunctive/DINO Boolean operator:
        Phi_{j,alpha}=1{there exists k with q_jk=alpha_k=1}.
    """
    Q_basis = np.asarray(Q_basis, dtype=np.int8)
    Jb, K = Q_basis.shape

    alphas = np.array(
        list(itertools.product([0, 1], repeat=K)),
        dtype=np.int8
    )

    if op == "conj":
        # Shape: (2^K, Jb). Each row is one Phi column transposed.
        phi_cols = (alphas[:, None, :] >= Q_basis[None, :, :]).all(axis=2)
    elif op == "disj":
        phi_cols = ((alphas[:, None, :] & Q_basis[None, :, :]).any(axis=2))
    else:
        raise ValueError("op must be either 'conj' or 'disj'.")

    return int(np.unique(phi_cols.astype(np.int8), axis=0).shape[0])


### main expr function

def identifiability_expr(Q, solver, op="conj", verbose=False):
    Q = Q.copy()

    diag = {
        "basis_time": 0.0,
        "trivial_check_time": 0.0,
        "two_col_check_time": 0.0,
        "three_col_check_time": 0.0,
        "identity_check_time": 0.0,
        "sat_time": 0.0,
    }

    def log(msg):
        if verbose:
            print(msg)

    t_alg_start = time.perf_counter()

    # Step 1: get Q_basis
    t = time.perf_counter()
    (
        Q_basis,
        basis_to_original,
        orig_indices_for_basis,
        Q_unique,
        unique_to_original,
        basis_to_unique
    ) = get_basis(Q)
    diag["basis_time"] = time.perf_counter() - t

    J_basis, K = Q_basis.shape
    diag["J_basis"] = int(J_basis)

    def finish(status, Q_bar, branch):
        # Algorithm time excludes extra diagnostic computations below.
        diag["algorithm_time"] = time.perf_counter() - t_alg_start

        # Extra diagnostics for tables. These should not be counted as algorithm runtime.
        t_diag = time.perf_counter()

        diag["identifiable"] = int(status)
        diag["branch"] = int(branch)
        diag["branch_label"] = BRANCH_LABELS.get(branch, "unknown")
        diag["sat_called"] = int(branch in (5, 6))

        # Original-Q diagnostics: use these for pure-node gap.
        diag["identity_original"] = int(contains_identity_submatrix(Q))
        diag["not_complete_original"] = int(not diag["identity_original"])
        diag["has_pure_node_original"] = int(has_any_pure_node(Q))
        diag["no_pure_nodes_original"] = int(not diag["has_pure_node_original"])
        diag["two_col_violation_original"] = int(violates_two_column_necessary(Q))

        # Basis-Q diagnostics: use these for algorithm/preprocessing behavior.
        diag["identity_basis"] = int(contains_identity_submatrix(Q_basis))
        diag["two_col_violation_basis"] = int(violates_two_column_necessary(Q_basis))
        diag["M_basis"] = count_representative_classes(Q_basis, op=op)

        diag["preprocess_time"] = (
            diag["basis_time"]
            + diag["trivial_check_time"]
            + diag["two_col_check_time"]
            + diag["three_col_check_time"]
            + diag["identity_check_time"]
        )

        diag["diagnostic_time"] = time.perf_counter() - t_diag

        return status, Q_bar, branch, diag

    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q_basis has J < K, thus not identifiable.")
        return finish(0, Q_bar, -1)

    # Step 3: trivial non-identifiability checks on Q_basis.
    t = time.perf_counter()
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            diag["trivial_check_time"] = time.perf_counter() - t
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            log("Q is trivially not identifiable: all-zero column.")
            return finish(0, Q_bar, 0)

        if np.all(Q_basis[:, k] == 1):
            diag["trivial_check_time"] = time.perf_counter() - t
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            log("Q is trivially not identifiable: all-one column.")
            return finish(0, Q_bar, 1)

    diag["trivial_check_time"] = time.perf_counter() - t

    # Two-column check.
    t = time.perf_counter()
    candidate = check_two_column_submatrices(Q_basis)
    diag["two_col_check_time"] = time.perf_counter() - t

    if candidate is not None:
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q is not identifiable: two-column submatrix not identifiable.")
        return finish(0, Q_bar, 2)

    # Three-column check.
    t = time.perf_counter()
    candidate = check_three_column_submatrices(Q_basis)
    diag["three_col_check_time"] = time.perf_counter() - t

    if candidate is not None:
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        log("Q is not identifiable: three-column submatrix not identifiable.")
        return finish(0, Q_bar, 3)

    # Identity-submatrix direct check.
    t = time.perf_counter()
    has_identity_basis = contains_identity_submatrix(Q_basis)
    diag["identity_check_time"] = time.perf_counter() - t

    if has_identity_basis:
        log("Q is identifiable by identity-submatrix direct check.")
        return finish(1, None, 4)

    # SAT step.
    t = time.perf_counter()

    if solver == -1:
        solution = solve_SAT_fast(Q_basis)

        diag["sat_time"] = time.perf_counter() - t

        if solution is not None:
            Q_basis_bar = solution
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return finish(0, Q_bar, 5)
        else:
            return finish(1, None, 6)

    else:
        Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)

        if solver == 0:
            solution = solve_SAT(Q_sorted, solver_name="cadical195")
        elif solver == 1:
            solution = solve_SAT(Q_sorted, solver_name="Glucose42")
        elif solver == 2:
            # This is currently identical to solver == 1 in your file.
            # Change this if solver == 2 is intended to mean something else.
            solution = solve_SAT(Q_sorted, solver_name="Glucose42")
        else:
            raise ValueError(f"Unknown solver code: {solver}")

        diag["sat_time"] = time.perf_counter() - t

        if solution is not None:
            Q_basis_bar = solution[:, sorted_to_original]
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return finish(0, Q_bar, 5)
        else:
            return finish(1, None, 6)
            
### running function            
def run_expr(J, K, N, p, seed, solver=-1, output_csv=None, op="conj"):
    """
    Randomly sample N binary matrices of shape (J, K) with entries drawn
    from Bernoulli(p). For each matrix, call identifiability_expr(Q)
    and record algorithm diagnostics.
    """
    RR = []
    np.random.seed(seed)

    if output_csv is None:
        output_csv = f"../data/raw/solver{solver}_J{J}_K{K}_p{p}_seed{seed}_diag.csv"

    outdir = os.path.dirname(output_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    for i in range(N):
        Q = np.random.binomial(1, p, size=(J, K))

        status, Q_bar, branch, diag = identifiability_expr(
            Q,
            solver=solver,
            op=op,
            verbose=False
        )

        results = {
            "J": J,
            "K": K,
            "N": N,
            "p": p,
            "seed": seed,
            "sim": i,
        }

        results.update(diag)

        file_exists = os.path.exists(output_csv)
        with open(output_csv, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

        RR.append(results)

    return RR


if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: python run_expr.py <J> <K> <N> <p> <seed> <solver>")
        sys.exit(1)

    J = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    p = float(sys.argv[4])
    seed = int(sys.argv[5])
    solver = int(sys.argv[6])
    
    RR = run_expr(J, K, N, p, seed, solver)

    print("Simulation results:")
    print(RR)
