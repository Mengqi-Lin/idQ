import os
import numpy as np

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


def load_identifiable_set(csv_file):
    """
    Load a set of canonical identifiable matrices from a CSV file.
    Each row in the CSV is assumed to contain a flattened binary string representation of a matrix.
    
    Returns:
        A set of strings.
    """
    identifiable_set = set()
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                identifiable_set.add(row[0].strip())
    return identifiable_set

def flatten_matrix(Q):
    """
    Flatten a binary matrix Q into a string representation.
    """
    return "".join(map(str, Q.flatten().tolist()))


def any_column_permutation_in_set(Q_reduced, identifiable_set):
    """
    Check if some column permutation of Q_reduced (with rows sorted lexicographically)
    is in identifiable_set.
    For small K, this iterates over all permutations.
    """
    J, K = Q_reduced.shape
    for perm in itertools.permutations(range(K)):
        Q_perm = Q_reduced[:, perm]
        # Sort the rows lexicographically to obtain a canonical representation.
        sorted_rows = sorted(Q_perm.tolist())
        # Flatten the sorted rows into a single string.
        flattened = "".join("".join(map(str, row)) for row in sorted_rows)
        if flattened in identifiable_set:
            return True
    return False



def direct_check_id(Q_reduced, csv_file=None):
    """
    Directly check if Q_reduced is identifiable.
    
    This function performs two checks:
      1. It checks if Q_reduced contains a permutation submatrix 
         (i.e., a KxK identity up to permutation).
      2. It checks whether Q_reduced (or any column permutation thereof) 
         matches one of the pre-enumerated identifiable matrices stored in a CSV file.
    
    If csv_file is not provided, it is automatically set based on the dimensions 
    of Q_reduced:
       For a Q_reduced of shape (J, K), the default file is "data/identifiable_Q_K{K}_J{J}.csv".
    
    Parameters:
        Q_reduced (np.ndarray): A binary matrix of shape (J, K).
        csv_file (str, optional): Path to a CSV file containing canonical representations of identifiable matrices.
    
    Returns:
        bool: True if Q_reduced is identifiable, False otherwise.
    """
    # Check 1: If Q_reduced contains a permutation submatrix (i.e., an identity matrix up to permutation)
    if contains_identity_submatrix(Q_reduced):
        return True
    
    J, K = Q_reduced.shape
    if csv_file is None:
        csv_file = f"data/identifiable_Q_K{K}_J{J}.csv"
    
    # If the CSV file does not exist, return False
    if not os.path.exists(csv_file):
        return False

    # Check 2: Load the pre-enumerated identifiable set from CSV and check against column permutations.
    identifiable_set = load_identifiable_set(csv_file)
    if any_column_permutation_in_set(Q_reduced, identifiable_set):
        return True

    return False
