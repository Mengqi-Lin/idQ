import itertools
import random
import numpy as np
from itertools import combinations, product, permutations
import warnings
import time
import networkx as nx
import matplotlib.pyplot as plt

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
    sorted_vectors = sorted(vectors, key=lambda x: sum((np.all(x <= y) and np.any(x < y)) for y in vectors), reverse=False)

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
            if not (np.all(q1 <= q2) or np.all(q1 >= q2)):
                # If q1 is parallel to q2, then q1_bar must be parallel to q2_bar
                if not (np.all(q1_bar <= q2_bar) or np.all(q1_bar >= q2_bar)):
                    return False

            # Check if q1 is strictly less than q2
            if np.all(q1 <= q2) and np.any(q1 < q2):
                # If q1 is strictly less than q2, then q1_bar must be strictly less than q2_bar or parallel to q2_bar
                if not (np.all(q1_bar <= q2_bar) and np.any(q1_bar < q2_bar)) and not (np.all(q1_bar <= q2_bar) or np.all(q1_bar >= q2_bar)):
                    return False

            # Check if q1 is strictly greater than q2
            if np.all(q2 <= q1) and np.any(q2 < q1):
                # If q1 is strictly greater than q2, then q1_bar must be strictly greater than q2_bar or parallel to q2_bar
                if not (np.all(q2_bar <= q1_bar) and np.any(q2_bar < q1_bar)) and not (np.all(q1_bar <= q2_bar) or np.all(q1_bar >= q2_bar)):
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

def column_rank_T_mat(Q):
    TT = T_mat(Q)
    return(np.linalg.matrix_rank(TT))


def global_identifiability(Q, uniout=True, check_level=3):
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

    # Check if Q has identical columns
    Q_T = Q.T
    Q_unique_cols = np.unique(Q_T, axis=0)
    if len(Q_unique_cols) != Q.shape[1]:
        print("Q is trivially not identifiable. Q contains identical columns.")
        # Find duplicate columns
        _, indices = np.unique(Q_T, return_index=True, axis=0)
        duplicate_columns = np.setdiff1d(np.arange(Q.shape[1]), indices)
        if len(duplicate_columns) > 0:
            Q_bar = Q.copy()
            Q_bar[:, duplicate_columns[0]] = 0  # Replace first duplicate column with all zeros
            return 0, [Q_bar]
    
    # Get the number of rows (J) and columns (K) of Q
    J, K = Q.shape

    # Calculate the distance to the universal set (d(Q[j,])) for each replaceable row
    distances = distances2U(Q)
#     print(distances)
    # Find out which rows are irreplaceable (having exactly one 1 or all 1)
    irreplaceable_rows = np.where(np.array(distances) == K-1)[0]
    
    # Find out which rows are replaceable
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    # Prepare the list to hold all possible Q_bar matrices
    Q_bars = []

    kappa = len(replaceable_rows)
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

    # After obtaining subQ_bars from the cartesian product
    for subQ_bar_replacements in subQ_bars:
        Q_bar = Q.copy()
        Q_bar[replacement_indices, :] = np.array(subQ_bar_replacements)  # Replace the rows in Q with subQ_bar_replacements

        # Check for preservation of partial order exactly among the rows specified by replacement_indices
        if preserve_partial_order(Q, Q_bar, replacement_indices, replacement_indices):
            Q_bars.append(Q_bar)  # If the partial order is preserved, add this Q_bar to the list

    # Now, Q_bars is a list of all possible Q_bar matrices that satisfy the conditions

    
    # Generate all permutations of Q
    Q_perms = generate_permutations(Q)

    
    # Filter out Q_bars that match a permutation of Q
    Q_bars = [Q_bar for Q_bar in Q_bars if not any(np.array_equal(Q_perm, Q_bar) for Q_perm in Q_perms)]
 
    if uniout:
        Q_bars_uniq = []
        seen = set()

        for Q_bar in Q_bars:
            # Generate all unique column permutations for each Q_bar
            perms_indices = list(itertools.permutations(range(Q_bar.shape[1])))

            # Rearrange columns according to the permutations, then convert to bytes
            perms_bytes = [Q_bar[:, p].tobytes() for p in perms_indices]

            # Check if the permutation has been seen before
            found = False
            for perm in perms_bytes:
                if perm in seen:
                    found = True
                    break
                seen.add(perm)
            if not found:
                Q_bars_uniq.append(Q_bar)

        Q_bars = Q_bars_uniq

    try:
        first_Q_bar = next(Q_bars)
    except StopIteration:
        print(f"Q is globally identifiable for check_level = 1.")
        return 1, []
    else:
        # If the generator is not empty, put the first element back and continue processing
        Q_bars = itertools.chain([first_Q_bar], Q_bars)

    if check_level == 1:
        print("Q might not be identifiable for check_level = 1, the possible Q_bars are: \n")
        return 0, [Q_bars]
    
    if check_level > 1:
        if check_level > max(distances):
            print("Warning: Check level is larger than max(distances).")
            check_level = max(distances)
        filtered_Q_bars = []
        for Q_bar in Q_bars:
            preserve_order = True
            for l in range(2, check_level + 1):
                D_l_Q = get_D_l(Q, l)
                D_l_Q_bar = get_D_l(Q_bar, l)

                JJ = D_l_Q.shape[0]
                if not preserve_partial_order(D_l_Q, D_l_Q_bar, range(JJ), range(JJ)):
                    preserve_order = False
                    break
            if preserve_order:
                filtered_Q_bars.append(Q_bar)
                
        if not filtered_Q_bars:
            print(f"Q is globally identifiable for check_level = {check_level}.")
            return 1, []    
        
        print(f"Q may not be identifiable for check_level = {check_level}, the possible Q_bars are: \n")
        return 0, filtered_Q_bars


### This function is an incomplete version of global_identifiability, if Q is not identifiable, it only returns one possible Q_bar.

def incomplete_global_identifiability(Q, uniout=True, check_level=3):
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

    # Check if Q has identical columns
    Q_T = Q.T
    Q_unique_cols = np.unique(Q_T, axis=0)
    if len(Q_unique_cols) != Q.shape[1]:
        print("Q is trivially not identifiable. Q contains identical columns.")
        # Find duplicate columns
        _, indices = np.unique(Q_T, return_index=True, axis=0)
        duplicate_columns = np.setdiff1d(np.arange(Q.shape[1]), indices)
        if len(duplicate_columns) > 0:
            Q_bar = Q.copy()
            Q_bar[:, duplicate_columns[0]] = 0  # Replace first duplicate column with all zeros
            return 0, [Q_bar]
    
    # Get the number of rows (J) and columns (K) of Q
    J, K = Q.shape

    # Calculate the distance to the universal set (d(Q[j,])) for each replaceable row
    distances = distances2U(Q)
#     print(distances)
    # Find out which rows are irreplaceable (having exactly one 1 or all 1)
    irreplaceable_rows = np.where(np.array(distances) == K-1)[0]
    
    # Find out which rows are replaceable
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    # Prepare the list to hold all possible Q_bar matrices
    Q_bars = []

    kappa = len(replaceable_rows)
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

    # After obtaining subQ_bars from the cartesian product
    def generate_Q_bars():
        for subQ_bar_replacements in subQ_bars:
            Q_bar = Q.copy()
            Q_bar[replacement_indices, :] = np.array(subQ_bar_replacements)  # Replace the rows in Q with subQ_bar_replacements

            # Check for preservation of partial order exactly among the rows specified by replacement_indices
            if preserve_partial_order(Q, Q_bar, replacement_indices, replacement_indices):
                yield Q_bar  # If the partial order is preserved, yield this Q_bar

    # Now, Q_bars is a generator of all possible Q_bar matrices that satisfy the conditions
    Q_bars = generate_Q_bars()
    
    # Generate all permutations of Q
    Q_perms = generate_permutations(Q)

    
    # Filter out Q_bars that match a permutation of Q
    Q_bars = (Q_bar for Q_bar in Q_bars if not any(np.array_equal(Q_perm, Q_bar) for Q_perm in Q_perms))

    
#     if uniout:
#         Q_bars_uniq = []
#         seen = set()

#         for Q_bar in Q_bars:
#             # Generate all unique column permutations for each Q_bar
#             perms_indices = list(itertools.permutations(range(Q_bar.shape[1])))

#             # Rearrange columns according to the permutations, then convert to bytes
#             perms_bytes = [Q_bar[:, p].tobytes() for p in perms_indices]

#             # Check if the permutation has been seen before
#             found = False
#             for perm in perms_bytes:
#                 if perm in seen:
#                     found = True
#                     break
#                 seen.add(perm)
#             if not found:
#                 Q_bars_uniq.append(Q_bar)

#         Q_bars = Q_bars_uniq

    try:
        first_Q_bar = next(Q_bars)
    except StopIteration:
        print(f"Q is globally identifiable for check_level = 1.")
        return 1, []
    else:
        # If the generator is not empty, put the first element back and continue processing
        Q_bars = itertools.chain([first_Q_bar], Q_bars)

    if check_level == 1:
        print("Q might not be identifiable for check_level = 1, the possible Q_bars are: \n")
        return 0, [first_Q_bar]
    
    if check_level > 1:
        if check_level > max(distances):
                    print("Warning: Check level is larger than max(distances).")
                    check_level = max(distances)
        for Q_bar in Q_bars:
            preserve_order = True
            for l in range(2, check_level + 1):
                D_l_Q = get_D_l(Q, l)
                D_l_Q_bar = get_D_l(Q_bar, l)

                JJ = D_l_Q.shape[0]
                if not preserve_partial_order(D_l_Q, D_l_Q_bar, range(JJ), range(JJ)):
                    preserve_order = False
                    break
            if preserve_order:
                print(f"Q may not be identifiable for check_level = {check_level}, the possible Q_bars are: \n")
                return 0, [Q_bar]
    
        print(f"Q is globally identifiable for check_level = {check_level}.")
        return 1, []                

## return typeid, Q_bars
## if typeid = 0, then not identifiable. typeid = 1, then identifiable. typeid = 2, trivially not identifiable.
def local_identifiability(Q, kappa, uniout=True, check_level=1):
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
            return 2, [Q_bar]
        if np.all(Q[:, k] == 1):
            Q_bar = Q.copy()
            Q_bar[:, k] = 0  # replace the k-th column with all zeros
            print("Q is trivially not identifiable. Q contains a one column.")
            return 2, [Q_bar]

    # Check if Q has identical columns
    Q_T = Q.T
    Q_unique_cols = np.unique(Q_T, axis=0)
    if len(Q_unique_cols) != Q.shape[1]:
        print("Q is trivially not identifiable. Q contains identical columns.")
        # Find duplicate columns
        _, indices = np.unique(Q_T, return_index=True, axis=0)
        duplicate_columns = np.setdiff1d(np.arange(Q.shape[1]), indices)
        if len(duplicate_columns) > 0:
            Q_bar = Q.copy()
            Q_bar[:, duplicate_columns[0]] = 0  # Replace first duplicate column with all zeros
            return 2, [Q_bar]
    
    # Get the number of rows (J) and columns (K) of Q
    J, K = Q.shape

    # Calculate the distance to the universal set (d(Q[j,])) for each replaceable row
    distances = distances2U(Q)
    
    # Find out which rows are irreplaceable (having exactly one 1 or all 1)
    irreplaceable_rows = np.where(np.array(distances) == K-1)[0]
    
    # Find out which rows are replaceable
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    if kappa > len(replaceable_rows):
        print("Q is kappa-locally identifiable because you can't replace kappa many of the rows.")
        return 1, []
    
    
    # Prepare the list to hold all possible Q_bar matrices
    Q_bars = []

    # Generate all permutations of Q
    Q_perms = generate_permutations(Q)
        
    # Iterate through all possible replacements of exactly kappa rows
    for replacement_indices in combinations(replaceable_rows, kappa):
        subQ_bars = []
        for i in range(kappa):
            index = replacement_indices[i]
            # Generate all possible replacements for the current set of rows,
            # binary vectors with at most K - distances[index] ones
            q_bars = []
            for p in range(1, K - distances[index] + 1):
                q_bars.extend(generate_binary_vectors(K, p))

            # Filter out the original row from q_bars
            q_bars = [q_bar for q_bar in q_bars if not np.array_equal(q_bar, Q[index])]

            # Filter out the rows that do not preserve the partial order
            Q_temp = Q.copy()
            valid_q_bars = []  # We will store the valid q_bars here

            # Iterate over all q_bars
            for q_bar in q_bars:
                Q_temp[index, :] = q_bar  # Replace the row in Q_temp
                
                fixed_rows = (set(replaceable_rows) - set(replacement_indices)) | set(irreplaceable_rows)
                # Check if the order is preserved after replacement
                if preserve_partial_order(Q, Q_temp, fixed_rows, [index]):
                    valid_q_bars.append(q_bar)  # If the order is preserved, keep this q_bar

            q_bars = valid_q_bars  # Replace the original q_bars with the valid ones

            subQ_bars.append(q_bars)

        # Generate Cartesian product of subQ_bars
        subQ_bars = list(product(*subQ_bars))

        # After obtaining subQ_bars from the cartesian product
        def generate_Q_bars():
            for subQ_bar_replacements in subQ_bars:
                Q_bar = Q.copy()
                Q_bar[replacement_indices, :] = np.array(subQ_bar_replacements)  # Replace the rows in Q with subQ_bar_replacements

                # Check for preservation of partial order exactly among the rows specified by replacement_indices
                if preserve_partial_order(Q, Q_bar, replacement_indices, replacement_indices):
                    yield Q_bar  # If the partial order is preserved, yield this Q_bar

        # Now, Q_bars is a generator of all possible Q_bar matrices that satisfy the conditions
        Q_bars_g = generate_Q_bars()

        # Filter out Q_bars that match a permutation of Q
        Q_bars_g = (Q_bar for Q_bar in Q_bars if not any(np.array_equal(Q_perm, Q_bar) for Q_perm in Q_perms))

        if uniout:
            Q_bars_uniq = []
            seen = set()

            for Q_bar in Q_bars_g:
                # Generate all unique column permutations for each Q_bar
                perms_indices = list(itertools.permutations(range(Q_bar.shape[1])))

                # Rearrange columns according to the permutations, then convert to bytes
                perms_bytes = [Q_bar[:, p].tobytes() for p in perms_indices]

                # Check if the permutation has been seen before
                found = False
                for perm in perms_bytes:
                    if perm in seen:
                        found = True
                        break
                    seen.add(perm)
                if not found:
                    Q_bars.append(Q_bar)
    
    try:
        first_Q_bar = next(Q_bars)
    except StopIteration:
        print(f"Q is globally identifiable for check_level = 1.")
        return 1, []
    else:
        # If the generator is not empty, put the first element back and continue processing
        Q_bars = itertools.chain([first_Q_bar], Q_bars)

    if check_level == 1:
        print("Q might not be identifiable for check_level = 1, the possible Q_bars are: \n")
        return 0, [Q_bars]
    
    if check_level > 1:
        if check_level > max(distances):
                    print("Warning: Check level is larger than max(distances).")
                    check_level = max(distances)
        filtered_Q_bars = []
        for Q_bar in Q_bars:
            preserve_order = True
            for l in range(2, check_level + 1):
                D_l_Q = get_D_l(Q, l)
                D_l_Q_bar = get_D_l(Q_bar, l)

                JJ = D_l_Q.shape[0]
                if not preserve_partial_order(D_l_Q, D_l_Q_bar, range(JJ), range(JJ)):
                    preserve_order = False
                    break
            if preserve_order:
                filtered_Q_bars.append(Q_bar)
                
        if not filtered_Q_bars:
            print(f"Q is {kappa}-locally identifiable.")
            return 1, []    
        
        print("Q may not be identifiable, the possible Q_bars are: \n")
        return 0, filtered_Q_bars
