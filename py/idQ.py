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
    return np.array(GG)

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


def fix_submatrix(Q, k1, k2):
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
            Q_bar_candidate = fix_submatrix(Q, k1, k2)
            if Q_bar_candidate is not None:
                return Q_bar_candidate
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

                               
                
### This function checks if Q is identifiable, if not, it returns one possible Q_bar.

def identifiability(Q):
    Q = Q.copy()
    # Check if Q is a binary matrix
    if not isinstance(Q, np.ndarray) or not np.all((Q == 0) | (Q == 1)):
        raise Exception("Q must be a binary matrix.")
    
    # Check if Q has all-zero rows and remove them
    row_sums = np.sum(Q, axis=1)
    if np.any(row_sums == 0):
        Q = Q[row_sums != 0]  # Keep only rows that are not all zeros
        print("All-zero rows have been removed from Q.")

    # Check if Q has identical rows
    Q_unique_rows = np.unique(Q, axis=0)
    if len(Q_unique_rows) != len(Q):
        Q = Q_unique_rows
        print("Removed identical rows of Q.")
        
    # Check if Q has all-zero or all-one columns
    for k in range(Q.shape[1]):
        if np.all(Q[:, k] == 0):
            Q_bar = Q.copy()
            Q_bar[:, k] = 1  # replace the k-th column with all ones
            print("Q is trivially not identifiable. Q contains a zero column.")
            return 0, [Q_bar]
        if np.all(Q[:, k] == 1):
            Q_bar = Q.copy()
            Q_bar[:, k] = 0  # replace the k-th column with all zeros
            print("Q is trivially not identifiable. Q contains a one column.")
            return 0, [Q_bar]

    Q_bar = check_two_column_submatrices(Q)
    if Q_bar is not None:
        return 0, [Q_bar]
    
    # Get the number of rows (J) and columns (K) of Q
    J, K = Q.shape

    # Calculate the distance to the universal set (d(Q[j,])) for each replaceable row
    distances = distances2U(Q)
    # Find out which rows are irreplaceable (having exactly one 1 or all 1)
    irreplaceable_rows = np.where(np.array(distances) == K-1)[0]
    
    # Find out which rows are replaceable
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    # Prepare the list to hold all possible Q_bar matrices
    Q_bars = []

    replacement_indices = list(replaceable_rows)  # since there is only one combination
    subQ_bars = []

    for i in range(len(replaceable_rows)):
        index = replacement_indices[i]
        # Generate all possible replacements for the current set of rows,
        # binary vectors with at most K - distances[index] ones
        q_bars = []
        for p in range(1, K - distances[index] + 1):
            q_bars.extend(generate_binary_vectors(K, p))

        # Filter out the rows that do not preserve the partial order
        Q_temp = Q.copy()
        valid_q_bars = []  # We will store the valid q_bars here

        # Iterate over all q_bars
        for q_bar in q_bars:
            Q_temp[index, :] = q_bar  # Replace the row in Q_temp
            # Check if the order is preserved after replacement
            if preserve_partial_order(Q, Q_temp, set(irreplaceable_rows), [index]):
                valid_q_bars.append(q_bar)  # If the order is preserved, keep this q_bar

        q_bars = valid_q_bars  # Replace the original q_bars with the valid ones
        subQ_bars.append(q_bars)

    # Generate Cartesian product of subQ_bars
    subQ_bars = itertools.product(*subQ_bars)
    
    # Generate unique Q_bar candidates (up to permutation equivalence).
    Q_bar_gen = generate_unique_Q_bars(subQ_bars, Q, replacement_indices)

    
    try:
        first_Q_bar = next(Q_bar_gen)
    except StopIteration:
        print("Q is identifiable")
        return 1, []
    else:
        Q_bar_gen = itertools.chain([first_Q_bar], Q_bar_gen)
    
    # Compute Phi(Q) for the original Q.
    PhiQ = Phi_mat(Q)  # User-provided function.
    
    # Iterate through all candidate Q_bar matrices.
    for Q_bar in Q_bar_gen:
        PhiB = Phi_mat(Q_bar)
        if thmCheck(PhiQ, PhiB, tol=1e-8):
            print("Q is not identifiable")
            return 0, Q_bar
    
    print("Q is identifiable")
    return 1, []
      
