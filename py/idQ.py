import itertools
import random
import numpy as np
from itertools import combinations, product, permutations
import warnings
import time
import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from solve_IP import solve_IP, solve_IP_fast
from solve_SAT import solve_SAT, solve_SAT_fast
from Qbasis import (
    get_basis, 
    get_Qunique_from_Qbasis, 
    get_Q_from_Qunique, 
    get_Q_from_Qbasis
)
import os


def contains_identity_submatrix(Q):
    """
    Checks if the binary matrix Q (shape JxK) has a subset of rows forming the identity matrix (size KxK).

    Parameters
    ----------
    Q : numpy.ndarray
        Binary matrix of size JxK to check for the presence of the identity matrix.

    Returns
    -------
    bool
        True if Q contains a subset of rows forming the identity matrix, False otherwise.
    """
    
    J, K = Q.shape

    # If J < K, it's not possible for Q to contain I_K
    if J < K:
        return False

    # Get the indices of pure_nodes in Q
    pure_nodes = np.where(np.sum(Q, axis=1) == 1)[0]
    
    # Get the unique pure_nodes
    pure_nodes_unique = np.unique(pure_nodes, axis=0)

    # Check if there are exactly K unique pure_nodes
    if len(pure_nodes_unique) == K:
        return True

    return False
                


def fix_submatrix2(Q, k1, k2):
    """
    Check the two-column submatrix of Q (columns k1 and k2) for a permutation matrix.
    If the submatrix does not contain at least one row (1,0) and one row (0,1), modify Q
    according to one of the three cases and return the modified Q (denoted Q_bar).
    If the submatrix already contains a permutation matrix, return None.
    
    Parameters:
        Q (np.ndarray): Binary matrix of shape (J, K).
        k1, k2 (int): Indices of the two columns to be checked.
        
    Returns:
        np.ndarray or None: Modified Q_bar if a change is needed (indicating non-identifiability),
                            or None if the submatrix is already valid.
    """
    # Extract the submatrix for columns k1 and k2.
    S = Q[:, [k1, k2]]
    # Get the unique rows as tuples.
    unique_rows = {tuple(row) for row in S}
    
    # Define the four possible row patterns.
    r00 = (0, 0)
    r10 = (1, 0)
    r01 = (0, 1)
    r11 = (1, 1)
    
    # If both (1,0) and (0,1) are present, no modification is needed.
    if r10 in unique_rows and r01 in unique_rows:
        return None
    
    # Create a copy to modify.
    Q_bar = Q.copy()
    
    # --- Case 1: Neither (1,0) nor (0,1) appears ---
    if r10 not in unique_rows and r01 not in unique_rows:
        # The only possible nonzero row is (1,1) (besides (0,0)).
        rows_r11 = np.where(np.all(S == np.array(r11), axis=1))[0]
        if len(rows_r11) > 0:
            # Modify one (1,1) row to be (1,0) (or (0,1); here we choose (1,0)).
            row_to_modify = rows_r11[0]
            Q_bar[row_to_modify, k1] = 1
            Q_bar[row_to_modify, k2] = 0
            return Q_bar
        else:
            # If the submatrix is entirely (0,0), modify the first row to (1,0).
            Q_bar[0, k1] = 1
            Q_bar[0, k2] = 0
            return Q_bar

    # --- Case 2: Submatrix only has (1,0) (but not (0,1)) ---
    if r10 in unique_rows and r01 not in unique_rows:
        # First, try to replace all (1,1) rows with (0,1) to introduce (0,1).
        rows_r11 = np.where(np.all(S == np.array(r11), axis=1))[0]
        if len(rows_r11) > 0:
            for row in rows_r11:
                Q_bar[row, k1] = 0
                Q_bar[row, k2] = 1
            return Q_bar
        else:
            # If no (1,1) exists, then all nonzero rows are (1,0). Replace all such rows with (0,1).
            rows_r10 = np.where(np.all(S == np.array(r10), axis=1))[0]
            for row in rows_r10:
                Q_bar[row, k1] = 0
                Q_bar[row, k2] = 1
            return Q_bar

    # --- Case 3: Submatrix only has (0,1) (but not (1,0)) ---
    if r01 in unique_rows and r10 not in unique_rows:
        # First, try to replace all (1,1) rows with (1,0) to introduce (1,0).
        rows_r11 = np.where(np.all(S == np.array(r11), axis=1))[0]
        if len(rows_r11) > 0:
            for row in rows_r11:
                Q_bar[row, k1] = 1
                Q_bar[row, k2] = 0
            return Q_bar
        else:
            # If no (1,1) exists, then all nonzero rows are (0,1). Replace all such rows with (1,0).
            rows_r01 = np.where(np.all(S == np.array(r01), axis=1))[0]
            for row in rows_r01:
                Q_bar[row, k1] = 1
                Q_bar[row, k2] = 0
            return Q_bar

    # Fallback (should not occur)
    return None

# Two-column submatrix check:
def check_two_column_submatrices(Q):
    """
    Iterate over all pairs of columns in Q and check whether each two-column submatrix contains a permutation matrix.
    If any pair fails the check, return a candidate Q_bar (modified Q) that fixes the deficiency.
    If all pairs are valid, return None.
    
    Parameters:
        Q (np.ndarray): Binary matrix of shape (J, K).
        
    Returns:
        np.ndarray or None: A candidate Q_bar that fixes at least one violating two-column submatrix,
                            or None if every two-column submatrix satisfies the permutation condition.
    """
    J, K = Q.shape
    for k1 in range(K):
        for k2 in range(k1+1, K):
            Q_bar_candidate = fix_submatrix2(Q, k1, k2)
            if Q_bar_candidate is not None:
                return Q_bar_candidate
    return None




def has_pure_node_K3(S):
    """
    Check if binary matrix S (list of lists or numpy array with shape (n,3))
    contains at least one pure node, i.e., a row equal to [1,0,0], [0,1,0], or [0,0,1].
    """
    try:
        S_arr = np.asarray(S)
        if S_arr.ndim != 2 or S_arr.shape[1] != 3:
            raise ValueError("Input must be a 2D array with three columns.")
        # A pure node in binary input has row-sum == 1
        return bool(np.any(S_arr.sum(axis=1) == 1))
    except ImportError:
        # Fallback to pure Python if numpy is unavailable
        unit_vectors = {(1,0,0), (0,1,0), (0,0,1)}
        for row in S:
            if tuple(row) in unit_vectors:
                return True
        return False


def fix_submatrix3(Q, k1, k2, k3):
    """
    Check the three-column submatrix of Q (columns k1, k2, k3).
    If the submatrix does not contain (up to a permutation of its columns)
    one of the desired patterns, attempt to fix it by forcing in one missing 
    standard basis vector.
    
    Parameters:
        Q (np.ndarray): Binary matrix of shape (J, K).
        k1, k2, k3 (int): Indices of the three columns to be checked.
        
    Returns:
        np.ndarray or None: A candidate Q_bar (modified Q) if a fix is applied,
                            or None if the submatrix is already valid.
    """
    S = Q[:, [k1, k2, k3]]
    if has_pure_node_K3(S):
        return None  # S is valid.
    
    # As a simple fix, enforce Condition 1: force in one missing standard basis vector.
    basis = {(1,0,0), (0,1,0), (0,0,1)}
    U = {tuple(row) for row in S}
    missing = basis - U
    Q_bar = Q.copy()
    if missing:
        missing_vector = list(missing)[0]
        # Try to find a row in S that is not already one of the basis vectors.
        for i in range(S.shape[0]):
            if tuple(S[i]) not in basis:
                Q_bar[i, [k1, k2, k3]] = np.array(missing_vector)
                return Q_bar
        # Fallback: modify the first row.
        Q_bar[0, [k1, k2, k3]] = np.array(missing_vector)
        return Q_bar
    return None

def check_three_column_submatrices(Q):
    """
    Iterate over all triples of columns in Q and check whether each 
    three-column submatrix (of the selected columns) contains one of the valid patterns 
    (up to a permutation of its columns). For each triple (k1, k2, k3), call fix_submatrix3(Q, k1, k2, k3).
    
    If any triple fails the check (i.e. fix_submatrix3 returns a candidate), return that candidate Q_bar.
    If all triples satisfy one of the valid patterns, return None.
    
    Parameters:
        Q (np.ndarray): Binary matrix of shape (J, K).
    
    Returns:
        np.ndarray or None: A candidate Q_bar that fixes at least one violating three-column submatrix,
                            or None if every triple is valid.
    """
    J, K = Q.shape
    for (k1, k2, k3) in itertools.combinations(range(K), 3):
        Q_bar = fix_submatrix3(Q, k1, k2, k3)
        if Q_bar is not None:
            return Q_bar
    return None



def canonicalize(Q):
    """
    Returns a canonical form of Q_bar by sorting its columns lexicographically.
    This canonical form is invariant under column permutations.
    """
    # Convert Q_bar to a list of tuples for each column
    cols = [tuple(col) for col in Q.T]
    # Sort the columns (this defines a canonical order)
    cols_sorted = sorted(cols)
    # Convert back to a numpy array (columns sorted)
    return np.array(cols_sorted).T  # Transpose back so that shape is (J, K)





def lex_sort_columns(Q: np.ndarray):
    """
    Return:
      - Q_sorted: Q with columns sorted in non-increasing lexicographic order
                  (row 0 is most-significant bit),
      - inv_perm: the mapping from Q_sorted's columns back to Q's columns,
                  i.e., Q[:, inv_perm] == Q_original.
    """
    J, K = Q.shape
    codes = []
    for k in range(K):
        col_bits = 0
        for j in range(J):
            if Q[j, k]:
                col_bits |= 1 << (J - 1 - j)
        codes.append((col_bits, k))  # (column code, original index)

    # Sort descending by column code, breaking ties by original index
    codes.sort(key=lambda t: (-t[0], t[1]))
    perm = [idx for (_, idx) in codes]  # new order
    Q_sorted = Q[:, perm]

    # Compute inverse permutation
    inv_perm = [0] * K
    for i, p in enumerate(perm):
        inv_perm[p] = i

    return Q_sorted, inv_perm
                
    


### This function checks if Q is identifiable, if not, it returns one possible Q_bar.
def identifiability(Q, solver = 0):
    """
    Check if Q is identifiable. If not, return one possible candidate Q_bar.
    
    The function proceeds as follows:
      1. Remove zero and identical rows from Q, yielding Q_unique and mapping_zero_identical.
      2. Remove rows that are generated by others from Q_unique, yielding Q_basis and mapping_generated.
      3. Perform trivial column checks and two-/three-column submatrix checks on Q_basis.
      4. Directly check identifiability of Q_basis using direct_check_id.
      5. If direct_check_id returns False, use brute_check_id on Q_basis.
      6. If a candidate Q_basis_bar is found, lift it back to a candidate Q_bar for the original Q.
    
    Returns:
        (status, Q_bar) where status == 1 indicates Q is identifiable,
        and status == 0 indicates Q is not identifiable and Q_bar is a candidate.
    """
    Q = Q.copy()
    
    # Step 1: get Q_basis
    (Q_basis, basis_to_original, orig_indices_for_basis,
         Q_unique, unique_to_original, basis_to_unique) = get_basis(Q)
    
    
    
    J_basis, K = Q_basis.shape
    
    if J_basis < K:
        Q_basis_bar = np.eye(K, dtype=int)[:J_basis]
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        print("Q_basis with dimensions J < K, thus not identifiable.")
        return 0, Q_bar
    # All subsequent candidate generation is performed on Q_basis.
    # Step 3: Check for trivial non-identifiability on Q_basis.
    for k in range(K):
        if np.all(Q_basis[:, k] == 0):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 1
            print("Q is trivially not identifiable (all zero column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        if np.all(Q_basis[:, k] == 1):
            Q_basis_bar = Q_basis.copy()
            Q_basis_bar[:, k] = 0
            print("Q is trivially not identifiable (all one column).")
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
    
    Q_basis_bar = check_two_column_submatrices(Q_basis)
    if Q_basis_bar is not None:
        print("Q is not identifiable (two-column submatrix not identifiable).")
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar
    elif K == 2:
        return True, None
    
    Q_basis_bar = check_three_column_submatrices(Q_basis)
    if Q_basis_bar is not None:
        print("Q is not identifiable (three-column submatrix not identifiable).")
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar
    elif K == 3:
        return True, None
    
    # Step 4: Determine identifiability of Q_basis.
    if contains_identity_submatrix(Q_basis):
        print("Q contains an identity submatrix, thus is identifiable.")
        return True, None
    else:
        # if fast:
        #     solution = solve_Q_identifiability_fast(Q_basis)
        #     if solution is not None:
        #         Q_basis_bar = solution
        #         Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        #         return 0, Q_bar
        #     else:
        #         return True, None  # No solution exists
        # else:
        Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)
        solution = solve_SAT(Q_sorted)
        if solution is not None:
            Q_basis_bar = solution[:, sorted_to_original]
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        else:
            return True, None  # No solution exists
        
        