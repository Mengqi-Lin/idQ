import itertools
import random
import numpy as np
from itertools import combinations, product, permutations
import warnings
import time
import networkx as nx
import matplotlib.pyplot as plt

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
    pp = [seq for seq in itertools.product([0, 1], repeat=K)]
    GG = []
    for j in range(J):
        GG.append((list(is_less_equal_than(Q[j], p) for p in pp)))
    return np.array(GG, dtype=int)


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
    Compute the representative node set R(Q) from the Q-matrix.
    
    Here, Q is a binary numpy array of shape (J, K). For each subset S of [J],
    we compute the bitwise OR (i.e., elementwise maximum) over the rows indexed by S.
    
    Formally, R(Q) = { v(⋁_{j in S} q_j) | S ⊆ [J] }.
    
    The function returns a set of tuples, each representing a unique vector.
    
    Parameters:
        Q (np.ndarray): A binary matrix of shape (J, K).
    
    Returns:
        set: A set of tuples, each tuple is a representative node.
    """
    J, K = Q.shape
    node_set = set()
    
    # Iterate over all subsets of indices from 0 to J (including the empty subset)
    for r in range(0, J + 1):
        for subset in itertools.combinations(range(J), r):
            if len(subset) == 0:
                # Define the OR of the empty set as the all-zero vector.
                or_vector = np.zeros(K, dtype=int)
            else:
                # Compute the elementwise OR of rows in the subset.
                or_vector = np.bitwise_or.reduce(Q[list(subset), :])
            node_set.add(tuple(or_vector))
    
    return node_set


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
    
def get_D_l(Q, l):
    # Get the total number of rows
    J = Q.shape[0]
    
    # Create an empty list to store the resulting vectors
    D_l = []
    
    # Generate all combinations of l rows
    for row_indices in combinations(range(J), l):
        # Select the rows specified by the current combination
        rows = Q[row_indices, :]
        
        # Perform the bitwise OR operation on the selected rows and add the result to D_l
        D_l.append(np.bitwise_or.reduce(rows, axis=0))
    
    # Convert D_l to a NumPy array
    D_l = np.array(D_l)
    
    return D_l

def get_D(Q, l):
    # Get the total number of rows
    J = Q.shape[0]
    D = set()
    for l in range(1, J+1):
        D_l = get_D(Q, l)
        D.add(D_l)
    
    return D


def generate_DAG(Q):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each row in Q
    current_layer_nodes = set(tuple(row) for row in Q)
    for node in current_layer_nodes:
        G.add_node(node)

    # Add nodes and edges for each layer
    for layer in range(2, Q.shape[0] + 1):
        next_layer_nodes = set()
        for node1, node2 in combinations(current_layer_nodes, 2):
            combined_vec = np.bitwise_or(node1, node2)
            G.add_node(tuple(combined_vec))
            if not np.array_equal(node1, combined_vec):
                G.add_edge(node1, tuple(combined_vec))
            if not np.array_equal(node2, combined_vec):
                G.add_edge(node2, tuple(combined_vec))
            next_layer_nodes.add(tuple(combined_vec))
        current_layer_nodes = next_layer_nodes
        
    return G


def height_of_Q(Q):
    # Create a directed graph
    G = generate_DAG(Q)
    return nx.dag_longest_path_length(G) 


def generate_hasse_diagram(Q):
    G = generate_DAG(Q)
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    plt.show()
    return G


def topo_order(matrix):
    # Get the list of row vectors
    vectors = [tuple(row) for row in matrix]

    # Sort the row vectors using the custom comparison function
    sorted_vectors = sorted(vectors, key=lambda x: sum(is_strictly_less_than(x, y) for y in vectors), reverse=False)

    # Yield the vectors in topologically sorted order. Can change to return sorted_vectors.
    for vector in sorted_vectors:
        yield vector
        
        
def distances2U(Q, draw = 0):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each row in Q
    qqs = [tuple(row) for row in Q]
    current_layer_nodes = set(qqs)
    for node in current_layer_nodes:
        G.add_node(node)

    # Add nodes and edges for each layer
    for layer in range(2, Q.shape[0] + 1):
        next_layer_nodes = set()
        for node1, node2 in combinations(current_layer_nodes, 2):
            combined_vec = np.bitwise_or(node1, node2)
            G.add_node(tuple(combined_vec))
            if not np.array_equal(node1, combined_vec):
                G.add_edge(node1, tuple(combined_vec))
            if not np.array_equal(node2, combined_vec):
                G.add_edge(node2, tuple(combined_vec))
            next_layer_nodes.add(tuple(combined_vec))
        current_layer_nodes = next_layer_nodes
    
    
    # Draw the graph
    if(draw):        
        TR = nx.transitive_reduction(G)
        pos = nx.spring_layout(TR)
        nx.draw_networkx(TR, pos)
        plt.show()

    source = np.max(Q, axis = 0)
    dist = {node: -1 for node in G.nodes}
    dist[tuple(source)] = 0
    
    # Process nodes in topological order
    for v in topo_order(G.nodes):
        # Update longest path for each adjacent node
        for u in G.pred[v]:  # successors of v
            if dist[v] + 1 > dist[u]:
                dist[u] = dist[v] + 1
    
    # Create a list of distances for nodes in qqs
    dist_qqs = [dist[node] for node in qqs]

    return dist_qqs


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


def check_for_identity(Q):
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

def Any_two_columns_contain_I(Q):
    """
    Checks if any two columns of the binary matrix Q (shape JxK) contain the 2x2 identity matrix.

    Parameters
    ----------
    Q : numpy.ndarray
        Binary matrix of size JxK to check.

    Returns
    -------
    bool
        True if any two columns of Q contain the 2x2 identity matrix, False otherwise.
    """
    J, K = Q.shape
    # Combinations of K choose 2 for column pairs
    column_combinations = combinations(range(K), 2)
    
    for col1, col2 in column_combinations:
        sub_matrix = Q[:, [col1, col2]]
        if not check_for_identity(sub_matrix):
            return False

    return True


                

def thmCheck(PhiQ, PhiB, tol=1e-8):
    """
    Check whether the columns of PhiQ are a subset of the columns of PhiB.
    
    That is, for each column v in PhiQ, there exists a column w in PhiB such that
    v and w are equal (within a tolerance, if necessary).

    Parameters:
        PhiQ (np.ndarray): The Phi matrix computed from Q, shape (m, n).
        PhiB (np.ndarray): The Phi matrix computed from an alternative candidate Q,
                           shape (m, p).
        tol (float): Tolerance for numerical equality (default 1e-8).

    Returns:
        bool: True if every column of PhiQ is found in PhiB, False otherwise.
    """
    m, n = PhiQ.shape
    m2, p = PhiB.shape
    if m != m2:
        raise ValueError("PhiQ and PhiB must have the same number of rows.")
    
    # Iterate over each column in PhiQ.
    for i in range(n):
        col_found = False
        colQ = PhiQ[:, i]
        # Check against each column in PhiB.
        for j in range(p):
            colB = PhiB[:, j]
            # Use np.allclose for floating point comparison, or np.array_equal if exact equality is expected.
            if np.allclose(colQ, colB, atol=tol):
                col_found = True
                break
        if not col_found:
            return False
    return True


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




def canonicalize(Q_bar):
    """
    Returns a canonical form of Q_bar by sorting its columns lexicographically.
    This canonical form is invariant under column permutations.
    """
    # Convert Q_bar to a list of tuples for each column
    cols = [tuple(col) for col in Q_bar.T]
    # Sort the columns (this defines a canonical order)
    cols_sorted = sorted(cols)
    # Convert back to a numpy array (columns sorted)
    return np.array(cols_sorted).T  # Transpose back so that shape is (J, K)




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
        if preserve_partial_order(Q, Q_bar, replacement_indices, replacement_indices):
            can_form = canonicalize(Q_bar)
            key = can_form.tostring()
            if key not in seen:
                seen.add(key)
                yield Q_bar


def remove_zero_identical(Q):
    """
    Given a binary matrix Q, remove all-zero rows and duplicate rows,
    while preserving the original information. This function returns a tuple (Q_unique, mapping),
    where Q_unique is the reduced matrix containing only unique nonzero rows, and mapping is a dictionary
    mapping each original row index to either 'zero' (if that row is all-zero) or an index in Q_unique.
    
    Parameters:
        Q (np.ndarray): Binary matrix of shape (J, K).
    
    Returns:
        Q_unique (np.ndarray): The reduced matrix with only unique nonzero rows.
        mapping (dict): A mapping from each original row index to either 'zero' or an integer index.
    """
    Q = np.array(Q)
    J, K = Q.shape
    
    mapping = {}
    nonzero_rows = []
    nonzero_orig_indices = []
    # Mark all-zero rows.
    for i in range(J):
        if np.all(Q[i] == 0):
            mapping[i] = 'zero'
        else:
            nonzero_rows.append(Q[i])
            nonzero_orig_indices.append(i)
    
    Q_nonzero = np.array(nonzero_rows)
    # Get unique rows among nonzero rows.
    unique_rows, unique_indices, inverse_indices = np.unique(Q_nonzero, axis=0, return_index=True, return_inverse=True)
    
    # Build mapping for nonzero rows.
    for idx, orig_idx in enumerate(nonzero_orig_indices):
        mapping[orig_idx] = int(inverse_indices[idx])
    
    return unique_rows, mapping



def lift_Q_zero_identical(Q, Q_unique_bar, mapping):
    """
    Reconstruct a full candidate Q_bar from a modified Q_unique_bar (the reduced matrix),
    using the mapping from remove_zero_identical. For each original row index i:
      - If mapping[i] is 'zero', then Q_bar[i] is the zero vector.
      - Otherwise, Q_bar[i] is set to Q_unique_bar[mapping[i]].
    
    Parameters:
        Q (np.ndarray): The original binary matrix (J x K).
        Q_unique_bar (np.ndarray): A modified version of the reduced matrix Q_unique.
        mapping (dict): Mapping from original row indices to either 'zero' or an index in Q_unique_bar.
    
    Returns:
        np.ndarray: The reconstructed full candidate matrix Q_bar (same shape as Q).
    """
    Q = np.array(Q)
    J, K = Q.shape
    Q_bar = np.zeros((J, K), dtype=Q.dtype)
    for i in range(J):
        if mapping[i] == 'zero':
            Q_bar[i] = np.zeros(K, dtype=Q.dtype)
        else:
            Q_bar[i] = Q_unique_bar[mapping[i]]
    return Q_bar




def remove_generated_rows(Q):
    """
    Remove rows of Q that can be generated by the bitwise OR of a nonempty subset of the other rows.
    Assumes Q has no identical or all-zero rows.
    
    Returns:
      (Q_reduced, mapping),
      where Q_reduced is the reduced matrix (subset of rows from Q)
      and mapping is a list of length J (the number of rows in Q).
      For row i in Q:
        - If row i is kept in Q_reduced, mapping[i] = [k], where k is its row index in Q_reduced.
        - If row i is generated by other rows j1, j2, ..., then mapping[i] = a list of row indices
          in Q_reduced corresponding to j1, j2, ... .
    """
    Q = np.array(Q)
    J, K = Q.shape
    keep = []
    # We initialize a mapping of length J to None. We'll fill it for each row i.
    mapping = [None] * J
    
    # 1) Identify generated vs. kept rows, but keep track of the generator subset
    #    in terms of the *original* indices
    generator_subsets = [None]*J  # generator_subsets[i] = a list of original row indices that generate row i
    
    for i in range(J):
        # If row i has norm 1, never generated. We'll keep it.
        if np.sum(Q[i]) == 1:
            keep.append(i)
            continue
        
        # Try to see if row i is generated by a nonempty subset of other rows.
        other_indices = [j for j in range(J) if j != i]
        generated = False
        generating_subset = None
        
        for r in range(1, len(other_indices)+1):
            for subset in itertools.combinations(other_indices, r):
                # Only consider subsets that are componentwise smaller than Q[i].
                if not all(np.all(Q[s] <= Q[i]) and np.any(Q[s] < Q[i]) for s in subset):
                    continue
                or_val = np.bitwise_or.reduce(Q[list(subset)], axis=0)
                if np.array_equal(or_val, Q[i]):
                    generated = True
                    generating_subset = list(subset)
                    break
            if generated:
                break
        
        if generated:
            generator_subsets[i] = generating_subset
        else:
            keep.append(i)
    
    # 2) Build Q_reduced from the kept rows
    Q_reduced = Q[keep, :]
    
    # 3) For each kept row i, we set mapping[i] = [k], where k is the new index in Q_reduced
    index_in_reduced = {}
    for new_idx, orig_idx in enumerate(keep):
        mapping[orig_idx] = [new_idx]
        index_in_reduced[orig_idx] = new_idx   # We'll use this dict for translation
    
    # 4) For each generated row i, we now translate the *original* subset of generator rows
    #    into Q_reduced indices
    for i in range(J):
        if mapping[i] is not None:
            continue  # already filled for a kept row
        # generator_subsets[i] is e.g. [j1, j2,...] in the original Q
        subset = generator_subsets[i]
        # Translate each j in subset to index_in_reduced[j]
        new_indices = []
        for j in subset:
            # j must have been kept or also a generated row => but if j is itself a generated row
            # we need a chain. Typically we assume a partial order so that j is kept or has a single "top-level" generator
            # but let's do a recursion or repeated translation for safety:
            new_indices.extend( _get_reduced_indices(j, mapping) )
        mapping[i] = sorted(set(new_indices))  # remove duplicates, sort for consistency
    
    return Q_reduced, mapping

def _get_reduced_indices(row_j, mapping):
    """
    Helper function: given row_j in Q, find the row(s) in Q_reduced that produce row_j.
    If mapping[row_j] = [k], that is a direct row in Q_reduced.
    If mapping[row_j] is a multi-element list, we gather all of those indices (or recursively expand).
    """
    if mapping[row_j] is None:
        return []
    # mapping[row_j] might be something like [k], or a list of multiple indices
    # but if multiple indices are themselves referencing others, we do a recursion
    # But in your linear scenario, you typically won't have multi-level generation. 
    # If you do, here's how you handle it:
    out = []
    stack = list(mapping[row_j])
    while stack:
        x = stack.pop()
        # if mapping[x] is single => a direct row, or multi => we need to expand 
        # but x is an index in the original Q. Typically we want to interpret x as a row in Q_reduced
        # The standard approach is: if len(mapping[x]) == 1 => direct index in Q_reduced
        # if len(mapping[x]) > 1 => chain
        # However, in a strict partial order, you won't get cycles.
        out.append(x)
    return out

def lift_Q_generated(Q, Q_reduced_bar, mapping):
    """
    Given the original Q, a modified Q_reduced_bar (of the same shape as Q_reduced),
    and the mapping from remove_generated_rows, reconstruct a full candidate Q_bar of 
    the same dimension as Q.
    
    For each row i in Q:
      - If mapping[i] = [k], then Q_bar[i] = Q_reduced_bar[k].
      - If mapping[i] = [k1, k2, ...], then Q_bar[i] = OR of Q_reduced_bar[k1], Q_reduced_bar[k2], ...
    """
    Q = np.array(Q)
    J, K = Q.shape
    Q_bar = np.zeros((J, K), dtype=Q.dtype)
    
    for i in range(J):
        indices = mapping[i]
        if len(indices) == 1:
            Q_bar[i] = Q_reduced_bar[indices[0]]
        else:
            Q_bar[i] = np.bitwise_or.reduce(Q_reduced_bar[indices], axis=0)
    return Q_bar





def direct_check_id(Q_reduced, csv_file=None):
    """
    Directly check if Q_reduced is identifiable.
    
    This function performs two checks:
      1. It checks if Q_reduced contains a permutation submatrix (i.e., a KxK identity up to permutation).
      2. It checks whether Q_reduced (or any column permutation thereof) matches one of the pre-enumerated
         identifiable matrices stored in a CSV file.
    
    If csv_file is not provided, it is automatically set based on the dimensions of Q_reduced:
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
    

    # Check 2: Load the pre-enumerated identifiable set from CSV and check against column permutations.
    identifiable_set = load_identifiable_set(csv_file)
    if any_column_permutation_in_set(Q_reduced, identifiable_set):
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
    Check if some column permutation of Q_reduced (canonicalized) is in identifiable_set.
    For small K, this iterates over all permutations.
    """
    J, K = Q_reduced.shape
    for perm in itertools.permutations(range(K)):
        Q_perm = Q_reduced[:, perm]
        # Optionally, we can sort rows to obtain a canonical representation.
        # For now, we simply flatten Q_perm.
        if flatten_matrix(Q_perm) in identifiable_set:
            return True
    return False
    
def brute_check_id(Q_reduced):
    """
    Perform candidate generation on the reduced matrix Q_reduced to check the identifiability.
    
    Q_reduced is the matrix obtained after removing all-zero, duplicate, and generated rows.
    
    The function generates candidate Q_reduced_bar matrices by modifying the replaceable rows of Q_reduced.
    It returns a tuple (status, Q_reduced_bar) where:
      - status == 0 indicates that Q_reduced is not identifiable and Q_reduced_bar is a candidate,
      - status == 1 indicates that Q_reduced is identifiable (and no candidate is found).
    
    Note: Lifting Q_reduced_bar back to the full Q will be handled in the main function.
    """
    J_reduced, K = Q_reduced.shape
    distances = distances2U(Q_reduced)
    irreplaceable_rows = np.where(np.array(distances) == K - 1)[0]
    replaceable_rows = set(range(J_reduced)) - set(irreplaceable_rows)
    
    replacement_indices = list(replaceable_rows)
    subQ_bars = []
    for i in range(len(replacement_indices)):
        index = replacement_indices[i]
        q_bars = []
        for p in range(1, K - distances[index] + 1):
            q_bars.extend(generate_binary_vectors(K, p))
        valid_q_bars = []
        Q_temp = Q_reduced.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q_reduced, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    # Generate Cartesian product of candidate replacements.
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q_reduced, replacement_indices)
    
    try:
        first_candidate = next(Q_bar_gen)
    except StopIteration:
        print("Q_reduced is identifiable.")
        return 1, None
    else:
        Q_bar_gen = itertools.chain([first_candidate], Q_bar_gen)
    
    PhiQ = Phi_mat(Q_reduced)
    for Q_reduced_bar in Q_bar_gen:
        PhiB = Phi_mat(Q_reduced_bar)
        if thmCheck(PhiQ, PhiB, tol=1e-8):
            print("Q_reduced is not identifiable.")
            return 0, Q_reduced_bar
    
    print("Q_reduced is identifiable.")
    return 1, None   
    
    


### This function checks if Q is identifiable, if not, it returns one possible Q_bar.
def identifiability(Q):
    """
    Check if Q is identifiable. If not, return one possible candidate Q_bar.
    
    The function proceeds as follows:
      1. Remove zero and identical rows from Q, yielding Q_unique and mapping_zero_identical.
      2. Remove rows that are generated by others from Q_unique, yielding Q_reduced and mapping_generated.
      3. Perform trivial column checks and two-/three-column submatrix checks on Q_reduced.
      4. Directly check identifiability of Q_reduced using direct_check_id.
      5. If direct_check_id returns False, use brute_check_id on Q_reduced.
      6. If a candidate Q_reduced_bar is found, lift it back to a candidate Q_bar for the original Q.
    
    Returns:
        (status, Q_bar) where status == 1 indicates Q is identifiable,
        and status == 0 indicates Q is not identifiable and Q_bar is a candidate.
    """
    Q = Q.copy()
    
    # Step 1: Remove zero and identical rows.
    Q_unique, mapping_zero_identical = remove_zero_identical(Q)
    print("Removed zero and identical rows. Q_unique shape:", Q_unique.shape)
    
    # Step 2: Remove generated rows.
    Q_reduced, mapping_generated = remove_generated_rows(Q_unique)
    print("Removed generated rows. Q_reduced shape:", Q_reduced.shape)
    
    # All subsequent candidate generation is performed on Q_reduced.
    # Step 3: Check for trivial non-identifiability on Q_reduced.
    for k in range(Q_reduced.shape[1]):
        if np.all(Q_reduced[:, k] == 0):
            Q_reduced_bar = Q_reduced.copy()
            Q_reduced_bar[:, k] = 1
            print("Q_reduced is trivially not identifiable (zero column).")
            Q_unique_bar = lift_Q_generated(Q_unique, Q_reduced_bar, mapping_generated)
            Q_bar = lift_Q_zero_identical(Q, Q_unique_bar, mapping_zero_identical)
            return 0, Q_bar
        if np.all(Q_reduced[:, k] == 1):
            Q_reduced_bar = Q_reduced.copy()
            Q_reduced_bar[:, k] = 0
            print("Q_reduced is trivially not identifiable (one column).")
            Q_unique_bar = lift_Q_generated(Q_unique, Q_reduced_bar, mapping_generated)
            Q_bar = lift_Q_zero_identical(Q, Q_unique_bar, mapping_zero_identical)
            return 0, Q_bar

    candidate = check_two_column_submatrices(Q_reduced)
    if candidate is not None:
        print("Identifiability determined at two-column check on Q_reduced.")
        Q_reduced_bar = candidate
        Q_unique_bar = lift_Q_generated(Q_unique, Q_reduced_bar, mapping_generated)
        Q_bar = lift_Q_zero_identical(Q, Q_unique_bar, mapping_zero_identical)
        return 0, Q_bar

    candidate = check_three_column_submatrices(Q_reduced)
    if candidate is not None:
        print("Identifiability determined at three-column check on Q_reduced.")
        Q_reduced_bar = candidate
        Q_unique_bar = lift_Q_generated(Q_unique, Q_reduced_bar, mapping_generated)
        Q_bar = lift_Q_zero_identical(Q, Q_unique_bar, mapping_zero_identical)
        return 0, Q_bar

    # Step 4: Determine identifiability of Q_reduced.
    if direct_check_id(Q_reduced):
        print("Q_reduced is directly identifiable (by direct_check).")
    else:
        status, Q_reduced_bar = brute_check_id(Q_reduced)
        if status == 0:
            print("Q_bar found via brute_check_id on Q_reduced.")
            Q_unique_bar = lift_Q_generated(Q_unique, Q_reduced_bar, mapping_generated)
            Q_bar = lift_Q_zero_identical(Q, Q_unique_bar, mapping_zero_identical)
            return 0, Q_bar
        else:
            print("Q_reduced is identifiable (by brute_check).")
    
    J, K = Q.shape
    distances = distances2U(Q)

    orig_mapping = original_indices_from_reduced(mapping_zero_identical, mapping_generated)
    # Flatten the mapping: union of all lists gives the set of irreplaceable original rows.
    irreplaceable_rows = set(sum(orig_mapping.values(), []))
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    replacement_indices = list(replaceable_rows)

    # Construct candidate set: all binary vectors of length K not in the representative set R(Q).
    all_vecs = list(itertools.product([0,1], repeat=Q.shape[1]))
    rep_set = representative_node_set(Q)  # Returns a set of tuples.
    cand_bars = [np.array(vec) for vec in all_vecs if tuple(vec) not in rep_set]

    subQ_bars = []
    for i in range(len(replacement_indices)):
        index = replacement_indices[i]
        q_bars = []
        norm_threshold = K - distances[index]
        q_bars = [cand for cand in cand_bars if np.sum(cand) <= norm_threshold]
        valid_q_bars = []
        Q_temp = Q.copy()
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar
            if preserve_partial_order(Q, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)
        subQ_bars.append(valid_q_bars)
    
    # Generate Cartesian product of candidate replacements.
    subQ_bars = itertools.product(*subQ_bars)
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q, replacement_indices)
    
    try:
        first_candidate = next(Q_bar_gen)
    except StopIteration:
        print("No candidate found in Q: Q is identifiable.")
        return 1, None
    else:
        Q_bar_gen = itertools.chain([first_candidate], Q_bar_gen)
    
    PhiQ = Phi_mat(Q)
    for Q_bar in Q_bar_gen:
        PhiB = Phi_mat(Q_bar)
        if thmCheck(PhiQ, PhiB, tol=1e-8):
            print("Q is not identifiable (candidate found).")
            return 0, Q_bar
    
    print("Q is identifiable.")
    return 1, None
    