import itertools
import random
import numpy as np
from itertools import combinations
from itertools import combinations, product
from itertools import permutations
import warnings
import time

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

def height_of_Q(Q):
    U_Q = np.bitwise_or.reduce(Q, axis=0)  # Compute the bitwise OR of all rows in Q

    # Iterate over the levels
    for l in range(1, Q.shape[0] + 1):
        D_l_Q = get_D_l(Q, l)  # Compute D^l(Q)

        # Get the number of unique rows in D_l_Q
        D_l_Q_unique_count = len(np.unique(D_l_Q, axis=0))

        # Check if the number of unique rows in D^l(Q) equals the number of rows in U(Q)
        if D_l_Q_unique_count == 1:
            return l - 1  # Return l - 1 if the condition is met

    return -1  # Return -1 if no such l is found


def distances2U(Q):
    J, _ = Q.shape  # Get the number of rows in Q
    distances = []
    if J == 1:
        return [0]
    
    for j in range(J):
        Q_new = []
        # Combine q_j with each of the other vectors that are not less or equal than q_j
        for i in range(J):
            if i != j and not is_less_equal_than(Q[i], Q[j]):
                Q_new.append(np.bitwise_or(Q[j], Q[i]))

        # Convert the list of new vectors to a numpy array
        Q_new = np.array(Q_new)

        # Check if Q_new is empty
        if Q_new.size == 0:
            distances.append(0)
            continue

        # Remove duplicate rows from Q_new
        Q_new = np.unique(Q_new, axis=0)

        # Calculate the height of the new Q
        height = height_of_Q(np.array(Q_new))
        
        # Add the height + 1 to the distances
        distances.append(height + 1)
    
    return distances



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

def local_identifiability(Q, kappa, uniout=False, check_level=1):
    # Check if Q is a binary matrix
    if not isinstance(Q, np.ndarray) or not np.all((Q == 0) | (Q == 1)):
        raise Exception("Q must be a binary matrix.")
    
    # Check if Q has all-zero rows and remove them
    row_sums = np.sum(Q, axis=1)
    if np.any(row_sums == 0):
        Q = Q[row_sums != 0]  # Keep only rows that are not all zeros
        print("All-zero rows have been removed from Q.")

    # Check if Q has all-zero or all-one columns
    for k in range(Q.shape[1]):
        if np.all(Q[:, k] == 0):
            Q_bar = Q.copy()
            Q_bar[:, k] = 1  # replace the k-th column with all ones
            return "Q is trivially not identifiable. Q contains a zero column.", [Q_bar]
        if np.all(Q[:, k] == 1):
            Q_bar = Q.copy()
            Q_bar[:, k] = 0  # replace the k-th column with all zeros
            return "Q is trivially not identifiable. Q contains a one column.", [Q_bar]

    # Check if Q has identical rows
    if np.unique(Q.view([('', Q.dtype)]*Q.shape[1]), return_index=True)[1].size != Q.shape[0]:
        print("Please remove identical rows of Q.")
        return [], []
    
    # Get the number of rows (J) and columns (K) of Q
    J, K = Q.shape
    
    # Find out which rows are irreplaceable (having exactly one 1 or all 1)
    irreplaceable_rows = np.where(np.sum(Q, axis=1) == 1)[0]
    
    # Find out which rows are replaceable
    replaceable_rows = set(range(J)) - set(irreplaceable_rows)
    
    if kappa > len(replaceable_rows):
        print("Q is kappa-locally identifiable because you can't replace kappa many of the rows.")
        return "Q is kappa-locally identifiable because you can't replace kappa many of the rows."
    
    # Calculate the distance to the universal set (d(Q[j,])) for each replaceable row
    distances = distances2U(Q)
    
    # Prepare the list to hold all possible Q_bar matrices
    Q_bars = []
    
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

    if check_level == 1:
        if not Q_bars:
            return f"Q is {kappa}-locally identifiable.", []
        return "Q may not be identifiable, the possible Q_bars are: \n", Q_bars
    
    if check_level > 1:
        if check_level >= J:
            warnings.warn("Check level is larger than J.")
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
            return f"Q is {kappa}-locally identifiable.", []    

        return "Q may not be identifiable, the possible Q_bars are: \n", filtered_Q_bars
