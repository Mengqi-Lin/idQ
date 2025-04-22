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
from functools import lru_cache
from solve_Q_identifiability import solve_Q_identifiability

from Qbasis import (
    get_basis, 
    get_Qunique_from_Qbasis, 
    get_Q_from_Qunique, 
    get_Q_from_Qbasis
)
import os
from direct_check_id import direct_check_id

# Compute T_matrix of Q
def T_mat(Q):
    J, K = Q.shape
    pp = [seq for seq in itertools.product([0, 1], repeat=K)]
    TT = []
    TT.append([1]*(2**K))
    for l in range(1, J+1):
        Dl = get_D_l(Q, l)
        for item in Dl:
            TT.append((list(is_less_equal_than(item, p) for p in pp)))
    return np.array(TT)

# Compute Phi_matrix of Q
def Phi_mat(Q):
    J, K = Q.shape
    AA = [seq for seq in itertools.product([0, 1], repeat=K)]
    GG = []
    for j in range(J):
        GG.append((list(is_less_equal_than(Q[j], aaa) for aaa in AA)))
    return np.array(GG, dtype=int)

def unique_response_columns(Q):
    """
    Computes the unique columns (response patterns) that would be in Phi_mat(Q) directly from Q.
    
    For each latent vector a in {0,1}^K, the response column is computed as:
        col = [ is_less_equal_than(Q[j], a) for j in range(J) ]
    where, for a binary Q and a, is_less_equal_than(Q[j], a) is True if 
    every 1 in Q[j] has the corresponding entry in a equal to 1.
    
    This function returns the set of unique columns as tuples of 0's and 1's.
    
    Parameters:
        Q (np.ndarray): A binary matrix of shape (J, K).
    
    Returns:
        set: A set of tuples, each representing a unique response column.
    """
    J, K = Q.shape
    unique_cols = set()
    # Iterate over all latent vectors a in {0,1}^K.
    for a in itertools.product([0, 1], repeat=K):
        # Convert a to a numpy array for vectorized comparisons.
        a_arr = np.array(a)
        response = np.all(Q <= a_arr, axis=1).astype(int)  # converts booleans to 0/1
        unique_cols.add(tuple(response))
    return unique_cols

def item_node_set(Q):
    """
    Returns the set of row vectors of Q.
    
    Parameters:
        Q (array-like): A matrix-like structure where each row represents a vector.
        
    Returns:
        set: A set containing each row vector of Q as a tuple.
    """
    return {tuple(row) for row in Q}


def representative_node_set(Q):
    """
    Build the representative node set R(Q) without enumerating all subsets.
    Instead, iteratively build up the set of distinct bitwise-OR combinations.

    Parameters:
        Q (np.ndarray): A J×K binary matrix.

    Returns:
        set: A set of length-K tuples corresponding to distinct bitwise-ORs
             of rows of Q.
    """
    J, K = Q.shape
    
    # Start with a set containing just the all-zero vector
    R = {tuple([0]*K)}
    
    # For each row in Q, OR it with every pattern in R (old or newly discovered)
    for row in Q:
        new_patterns = []
        for pattern in R:
            # Convert pattern (tuple) and row (ndarray) into an OR-combination
            or_vec = [pattern[i] | row[i] for i in range(K)]
            new_patterns.append(tuple(or_vec))
        
        # Add them to R
        for pat in new_patterns:
            R.add(pat)
    
    return R



def column_rank_T_mat(Q):
    TT = T_mat(Q)
    return(np.linalg.matrix_rank(TT))



# This function checks whether two given rows are parallel.
def is_parallel(row1, row2):
    return not (np.all(row1 <= row2) or np.all(row1 >= row2))

# This function checks whether row1 is strictly less than row2.
def is_strictly_less_than(row1, row2):
    return np.all(row1 <= row2) and np.any(row1 < row2)

# This function checks whether row1 is less than or equal to row2.
def is_less_equal_than(row1, row2):
    return np.all(row1 <= row2)

# This function generates all binary vectors of length K with a specified number of ones (num_of_ones).
def generate_binary_vectors(K, num_of_ones):
    binary_vectors = []
    for combination in combinations(range(K), num_of_ones):
        binary_vector = np.zeros(K, dtype=int)
        binary_vector[list(combination)] = 1
        binary_vectors.append(binary_vector)
    return binary_vectors

# This function generates all possible permutations of the columns in a 2D numpy array Q.
def generate_permutations(Q):
    K = Q.shape[1]  # number of columns
    Q_perms = []
    for perm in permutations(range(K)):  # generate all permutations of column indices
        Q_perm = Q[:, perm]  # apply permutation to columns of Q
        Q_perms.append(Q_perm)
    return Q_perms


# mask is just the integer version of a binary vector — optimized for speed and simplicity.
def row_masks(Q):
    """Return integer bit‑masks for each row of Q."""
    J, K = Q.shape
    masks = []
    for j in range(J):
        mask = 0
        for k in range(K):
            if Q[j, k]:
                mask |= 1 << k
        masks.append(mask)
    return masks, (1 << K) - 1  # list, full‑ones mask

def distances(Q):
    masks, full = row_masks(Q)
    J = len(masks)
    
    # lru_cache memoises the depth function, ensuring each reachable pattern is explored only once.
    @lru_cache(maxsize=None)
    def depth(mask):
        if mask == full:
            return 0
        best = 0
        for m in masks:
            new = mask | m
            if new != mask:                     # strict cover
                best = max(best, 1 + depth(new))
        return best

    return np.array([depth(m) for m in masks])


def preserve_partial_order(Q, Q_bar, indices1, indices2):
    for index1 in indices1:
        for index2 in indices2:
            q1, q2 = Q[index1], Q[index2]
            q1_bar, q2_bar = Q_bar[index1], Q_bar[index2]

            # Check if q1 is parallel to q2
            if is_parallel(q1, q2):
                # If q1 is parallel to q2, then q1_bar must be parallel to q2_bar
                if not is_parallel(q1_bar, q2_bar):
                    return False

            # Check if q1 is strictly less than q2
            if is_strictly_less_than(q1, q2):
                # If q1 is strictly less than q2, then q1_bar must be strictly less than q2_bar or parallel to q2_bar
                if not is_strictly_less_than(q1_bar, q2_bar) and not is_parallel(q1_bar, q2_bar):
                    return False

            # Check if q1 is strictly greater than q2
            if is_strictly_less_than(q2, q1):
                # If q1 is strictly greater than q2, then q1_bar must be strictly greater than q2_bar or parallel to q2_bar
                if not is_strictly_less_than(q2_bar, q1_bar) and not is_parallel(q1_bar, q2_bar):
                    return False

    # If none of the conditions are violated for any of the indices, return True
    return True


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
    Q = np.unique(Q, axis = 0)
    J, K = Q.shape

    # If J < K, it's not possible for Q to contain I_K
    if J < K:
        return False

    # Get the indices of rows in Q that contain exactly one 1
    single_one_rows = np.where(np.sum(Q, axis=1) == 1)[0]

    # Extract the rows that contain exactly one 1, and sort them in lexicographic order 
    Q_one = Q[single_one_rows]
    
    # Get the unique rows
    single_one_rows_unique = np.unique(single_one_rows, axis=0)

    # Check if there are exactly K unique rows
    if len(single_one_rows_unique) == K:
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



def valid_three_column_submatrix(S):
    """
    Given a three-column submatrix S (of shape (J,3)), return True if there exists 
    a permutation of its columns so that the unique rows of S contain one of the following target sets:
      1. The standard basis: {(1,0,0), (0,1,0), (0,0,1)}.
      2. The set T2: {(1,1,0), (1,0,1), (0,1,1), (1,0,0)}.
      3. The set T3: {(0,1,0), (0,0,1), (1,1,0), (1,0,1)}.
    Otherwise, return False.
    """
    # Unique rows of S as tuples.
    U = {tuple(row) for row in S}
    T1 = {(1,0,0), (0,1,0), (0,0,1)}
    T2 = {(1,1,0), (1,0,1), (0,1,1), (1,0,0)}
    T3 = {(0,1,0), (0,0,1), (1,1,0), (1,0,1)}
    
    # Try every permutation of columns.
    for perm in itertools.permutations(range(3)):
        U_perm = {tuple(row[i] for i in perm) for row in S}
        if T1.issubset(U_perm) or T2.issubset(U_perm) or T3.issubset(U_perm):
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
    if valid_three_column_submatrix(S):
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
        candidate = fix_submatrix3(Q, k1, k2, k3)
        if candidate is not None:
            return candidate
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


def generate_unique_Q_bars(subQ_bars, Q, replacement_indices):
    
    """
    Generate candidate Q_bar matrices from the Cartesian product subQ_bars,
    yielding only one representative per permutation equivalence class.
    The canonical form of Q is added to the 'seen' set so that any candidate
    equivalent to Q is automatically filtered out.
    
    Parameters:
        subQ_bars (iterable): Cartesian product of candidate replacement rows.
        Q (np.ndarray): The input Q-matrix.
        replacement_indices (list): Indices of rows in Q to be replaced.
    
    Yields:
        np.ndarray: A candidate Q_bar that is not permutation equivalent to Q.
    """
    
    seen = set()
    canonical_Q = canonicalize(Q)
    seen.add(canonical_Q.tostring())
    
    for subQ_bar_replacements in subQ_bars:
        # Check if the candidate for replacements is empty:
        if len(subQ_bar_replacements) == 0:
            continue  # skip this candidate
        
        Q_bar = Q.copy()
        candidate = np.array(subQ_bar_replacements)
        # Ensure candidate has the correct shape. If candidate.ndim == 1, reshape it.
        if candidate.ndim == 1:
            candidate = candidate.reshape(1, -1)
        
        Q_bar[replacement_indices, :] = candidate
        can_form = canonicalize(Q_bar)
        key = can_form.tostring()
        if key not in seen:
            seen.add(key)
            yield Q_bar


def thmCheck(Q_basis, Q_bar_gen):
    """
    For a given Q_basis and a candidate generator Q_bar_gen, verify for each candidate Q_basis_bar
    whether every unique response column S from Phi(Q_basis) is "matched" in Q_basis_bar.
    
    The check is as follows:
      For each S in unique_response_columns(Q_basis):
        - Let idx_active be the indices where S is 1.
        - Compute h_a as the logical OR of the rows Q_basis_bar[idx_active].
        - For each row j not in idx_active, check that Q_basis_bar[j] is not covered by h_a.
    
    Returns:
        (status, candidate)
          - status == 0 and candidate == Q_basis_bar if a candidate is found,
          - (1, None) if no candidate passes the check.
    """
    J_basis, K = Q_basis.shape
    cols_phiQ = unique_response_columns(Q_basis)
    
    for Q_basis_bar in Q_bar_gen:
        candidate_valid = True
        for S in cols_phiQ:
            S_arr = np.array(S, dtype=int)
            idx_active = [j for j in range(J_basis) if S_arr[j] == 1]
            
            # Compute h_a as the logical OR of rows Q_basis_bar for indices in idx_active.
            if idx_active:
                h_a = Q_basis_bar[idx_active[0]].copy()
                for j in idx_active[1:]:
                    h_a = np.logical_or(h_a, Q_basis_bar[j]).astype(int)
            else:
                h_a = np.zeros(K, dtype=int)
            
            # For any row j not in idx_active, check if Q_basis_bar[j] is "covered" by h_a.
            for j in range(J_basis):
                if j in idx_active:
                    continue
                if np.all(Q_basis_bar[j] <= h_a):
                    candidate_valid = False
                    break  # No need to check further S for this candidate.
            if not candidate_valid:
                break  # Move to next candidate Q_basis_bar.
        
        if candidate_valid:
            return 0, Q_basis_bar
    return 1, None
                
    
def brute_check_id(Q_basis):
    """
    Perform candidate generation on the basis matrix Q_basis to check the identifiability.
    
    Q_basis is the matrix obtained after removing all-zero, duplicate, and generated rows.
    
    The function generates candidate Q_basis_bar matrices by modifying the replaceable rows of Q_basis.
    It returns a tuple (status, Q_basis_bar) where:
      - status == 0 indicates that Q_basis is not identifiable and Q_basis_bar is a candidate,
      - status == 1 indicates that Q_basis is identifiable (and no candidate is found).
    
    Note: Lifting Q_basis_bar back to the full Q will be handled in the main function.
    """
    J_basis, K = Q_basis.shape
    distances = distances2U(Q_basis)
    irreplaceable_rows = np.where(np.array(distances) == K - 1)[0]
    replaceable_rows = set(range(J_basis)) - set(irreplaceable_rows)
    
    replacement_indices = list(replaceable_rows)
    subQ_bars = []
    for i in range(len(replacement_indices)):
        index = replacement_indices[i]
        q_bars = []
        # for p in range(1, K - distances[index] + 1):
        #     q_bars.extend(generate_binary_vectors(K, p))
        support = np.where(Q_basis[index, :] == 1)[0]
        for r in range(1, len(support) + 1):         # r = size of the subset, from 1 up to full support
            for combo in itertools.combinations(support, r):
                q = np.zeros(K, dtype=int)
                q[list(combo)] = 1
                q_bars.append(q)
        
        valid_q_bars = []
        Q_temp = Q_basis.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q_basis, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    # Generate Cartesian product of candidate replacements.
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q_basis, replacement_indices)

    return thmCheck(Q_basis, Q_bar_gen)

    


### This function checks if Q is identifiable, if not, it returns one possible Q_bar.
def identifiability(Q):
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

    candidate = check_two_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (two-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    candidate = check_three_column_submatrices(Q_basis)
    if candidate is not None:
        print("Q is not identifiable (three-column submatrix not identifiable).")
        Q_basis_bar = candidate
        Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
        return 0, Q_bar

    
    # Step 4: Determine identifiability of Q_basis.
    if direct_check_id(Q_basis):
        print("Q is identifiable (direct_check).")
        return 1, None
    else:
        Q_sorted, sorted_to_original = lex_sort_columns(Q_basis)
        solution = solve_Q_identifiability(Q_sorted)
        if solution is not None:
            Q_basis_bar = solution[:, sorted_to_original]
            Q_bar = get_Q_from_Qbasis(Q_basis_bar, basis_to_original)
            return 0, Q_bar
        else:
            return True, None  # No solution exists
        
        
    