# binary_matrix_reduction.py
import numpy as np
from collections import defaultdict

def get_basis(Q):
    """
    Takes a binary matrix Q (list of lists or a 2D NumPy array of 0/1) and returns:

      1) Q_basis:
         Final basis set of distinct, non-zero rows from Q (as a NumPy array),
         in lexicographic order, where any row that can be formed by OR of
         previously included rows is excluded.

      2) basis_to_original:
         A list (length = number of rows in original Q). For each row i in the original Q,
         basis_to_original[i] is a list of indices in Q_basis whose bitwise OR yields Q[i].

      3) original_indices_for_basis:
         For each row j in Q_basis, a list of the original Q row indices that exactly equal Q_basis[j].

      4) Q_unique:
         All distinct, non-zero rows in Q, sorted lexicographically, returned as a NumPy array.
         (Only duplicates and the all-zero row are removed; no OR-based elimination is performed.)

      5) unique_to_original:
         A list (length = number of rows in original Q) where unique_to_original[i] is the index in Q_unique
         that matches Q[i] exactly (or -1 if Q[i] is a zero row).

      6) basis_to_unique:
         A list (length = number of rows in Q_unique) where basis_to_unique[u] is a list of indices in Q_basis
         whose bitwise OR equals Q_unique[u].
    """
    # Ensure Q is a list of lists (if passed as a NumPy array, convert to list-of-lists)
    if isinstance(Q, np.ndarray):
        Q = Q.tolist()
    
    n = len(Q)
    if n == 0:
        return (np.array([]), [], [], np.array([]), [], [])
    
    # Convert each row of Q to a tuple so we can hash them.
    rows = [tuple(row) for row in Q]
    n_cols = len(rows[0])
    row_to_indices = defaultdict(list)
    for i, r in enumerate(rows):
        row_to_indices[r].append(i)

    zero_tuple = (0,) * n_cols

    # 1) Build Q_unique: all distinct, non-zero rows from Q, sorted lexicographically.
    all_tuples_sorted = sorted(row_to_indices.keys())  # includes zero row if present
    unique_tuples = [r for r in all_tuples_sorted if r != zero_tuple]
    # Convert to a NumPy array later; for now we keep them as tuples.
    Q_unique_list = [list(r) for r in unique_tuples]
    
    # Build a mapping from row tuple to its index in Q_unique.
    tuple_to_unique_idx = {r: idx for idx, r in enumerate(unique_tuples)}

    # 2) Build unique_to_original: for each row in original Q, the index in Q_unique (or -1 if zero).
    unique_to_original = [-1] * n
    for i, r in enumerate(rows):
        if r != zero_tuple:
            unique_to_original[i] = tuple_to_unique_idx[r]
        else:
            unique_to_original[i] = -1

    # 3) Build Q_basis from Q_unique using OR-based elimination.
    # or_dict maps any row (as a tuple) that can be formed by OR-ing Q_basis rows
    # to the list of indices (into Q_basis) used to form it.
    or_dict = {}
    if zero_tuple in row_to_indices:
        or_dict[zero_tuple] = []
    
    Q_basis_tuples = []
    original_indices_for_basis = []

    def bitwise_or(a, b):
        return tuple(x or y for x, y in zip(a, b))

    for r in unique_tuples:
        if r not in or_dict:
            # r is not yet OR-generated, so we add it to Q_basis.
            Q_basis_tuples.append(r)
            new_idx = len(Q_basis_tuples) - 1
            original_indices_for_basis.append(row_to_indices[r])
            # Immediately add r to or_dict.
            or_dict[r] = [new_idx]
            # Now update or_dict with combinations including r.
            existing_keys = list(or_dict.keys())
            for x in existing_keys:
                new_or = bitwise_or(x, r)
                if new_or not in or_dict:
                    or_dict[new_or] = or_dict[x] + [new_idx]

    Q_basis_list = [list(t) for t in Q_basis_tuples]

    # 4) Build basis_to_original: for each original row, indices in Q_basis whose OR yields it.
    basis_to_original = [or_dict[r] for r in rows]

    # 5) Build basis_to_unique:
    # For each row u in Q_unique, a list of Q_basis indices that OR together yield Q_unique[u].
    basis_to_unique = []
    for r in unique_tuples:
        basis_to_unique.append(or_dict[r])

    # Convert Q_basis_list and Q_unique_list to NumPy arrays.
    Q_basis = Q_basis = np.atleast_2d(np.array(Q_basis_list, dtype=int))
    Q_unique = np.array(Q_unique_list, dtype=int)

    return (
        Q_basis,                    # as a NumPy array
        basis_to_original,          # list of lists
        original_indices_for_basis, # list of lists
        Q_unique,                     # as a NumPy array
        unique_to_original,           # list
        basis_to_unique             # list of lists
    )

def get_Qunique_from_Qbasis(Q_basis, basis_to_unique):
    """
    Reconstruct Q_unique from Q_basis using basis_to_unique.

    :param Q_basis: A NumPy array of rows (each row a list of 0/1).
    :param basis_to_unique: List of lists, where each element is a list of indices in Q_basis 
                              whose bitwise OR yields a row in Q_unique.
    :return: Q_unique as a NumPy array.
    """
    if Q_basis.size == 0:
        return np.array([[]], dtype=int)
    n_cols = Q_basis.shape[1]
    Q_unique_list = []
    for subset in basis_to_unique:
        row = [0] * n_cols
        for ridx in subset:
            row = [row[c] | Q_basis[ridx, c] for c in range(n_cols)]
        Q_unique_list.append(row)
    return np.array(Q_unique_list, dtype=int)

def get_Q_from_Qunique(Q_unique, unique_to_original):
    """
    Reconstruct the original Q from Q_unique using unique_to_original.

    :param Q_unique: A NumPy array of rows (each row a list of 0/1).
    :param unique_to_original: List (length = number of rows in original Q) where unique_to_original[i] 
                               is the index in Q_unique that matches Q[i] (or -1 if Q[i] is zero).
    :return: The original Q as a NumPy array.
    """
    n = len(unique_to_original)
    if Q_unique.size == 0:
        return np.array([[]], dtype=int)
    n_cols = Q_unique.shape[1]
    Q_reconstructed = []
    for i in range(n):
        u = unique_to_original[i]
        if u == -1:
            row = [0] * n_cols
        else:
            row = Q_unique[u].tolist()  # copy the row from Q_unique
        Q_reconstructed.append(row)
    return np.array(Q_reconstructed, dtype=int)

def get_Q_from_Qbasis(Q_basis, basis_to_original):
    """
    Reconstruct the original Q from Q_basis using basis_to_original.

    :param Q_basis: A NumPy array of rows (each row a list of 0/1).
    :param basis_to_original: List (length = number of rows in original Q) where basis_to_original[i]
                                is a list of indices in Q_basis whose OR yields Q[i].
    :return: The original Q as a NumPy array.
    """
    n = len(basis_to_original)
    if Q_basis.size == 0:
        return np.array([[]], dtype=int)
    n_cols = Q_basis.shape[1]
    Q_reconstructed = []
    for subset in basis_to_original:
        row = [0] * n_cols
        for ridx in subset:
            row = [row[c] | Q_basis[ridx, c] for c in range(n_cols)]
        Q_reconstructed.append(row)
    return np.array(Q_reconstructed, dtype=int)
