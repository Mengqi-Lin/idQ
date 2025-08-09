import numpy as np
import itertools

def canonicalize(Q):
    """
    Returns the canonical form of Q by sorting its columns lexicographically.
    This canonical form is invariant under column permutation.
    
    Parameters:
        Q (np.ndarray): A binary matrix of shape (J, K).
        
    Returns:
        np.ndarray: The canonical form of Q.
    """
    # Convert each column into a tuple, sort them, and rebuild the matrix.
    cols = [tuple(col) for col in Q.T]
    cols_sorted = sorted(cols)
    return np.array(cols_sorted).T


## Randomly generate J times K binary matrix.
def random_generate_Q(J, K):
    # Initialize an empty set to store unique sequences
    sequences = set()

    # Loop until J unique sequences are found
    while len(sequences) < J:
        # Generate a random binary sequence of length K
        seq = tuple(np.random.choice([0, 1], size=K))
        
        # If the sequence is not all zeros, add it to the set
        if any(seq):
            sequences.add(seq)
    
    # Convert the selected sequences into a NumPy array
    Q = np.array(list(sequences))

    # Sort rows of Q
    Q = Q[np.lexsort(Q.T[::-1])]

    return Q

def binary_matrix_to_string(binary_matrix):
    return '_'.join(''.join(map(str, row)) for row in binary_matrix)



def sort_binary_matrix(matrix, columnfirst = 1):
    if not columnfirst:
        matrix = matrix.T
    # Calculate the column norms
    col_norms = np.sum(matrix, axis=0)
    
    # Sort columns based on their norms
    sorted_col_index = np.argsort(col_norms)
    matrix_sorted = matrix[:, sorted_col_index]
    
    # Calculate the new row norms
    row_norms_sorted = np.sum(matrix_sorted, axis=1)
    
    # Combine the row norms and the lexicographical order as the sorting keys
    lexico_rows = ["".join(map(str, row)) for row in matrix_sorted.tolist()]
    sorting_keys = sorted(range(len(lexico_rows)), key=lambda i: (row_norms_sorted[i], lexico_rows[i]))
    
    # Sort rows based on their norms, and then lexicographically for rows with equal norms
    matrix_sorted = matrix_sorted[sorting_keys, :]
    
    if not columnfirst:
        return matrix_sorted.T
    
    return matrix_sorted



def sort_lexicographically(matrix, axis):
    if axis == 0:  # sorting columns
        # Transpose matrix for column-wise operations
        matrix = matrix.T

    # Convert binary rows to integers for sorting
    int_rows = [int(''.join(map(str, row)), 2) for row in matrix]

    # Get the indices that would sort the rows
    sorted_indices = np.argsort(int_rows)

    # Sort the matrix using the sorted indices
    sorted_matrix = matrix[sorted_indices]

    if axis == 0:  # sorting columns
        # Transpose matrix back to original shape
        sorted_matrix = sorted_matrix.T

    return sorted_matrix


def generate_canonical_matrices(J, K):
    # Generate all possible binary sequences of length K (excluding all zeros)
    all_sequences = [seq for seq in itertools.product([0, 1], repeat=K) if any(seq)]
    all_canonical_matrices = [] # This will hold the actual matrices
    canonical_matrices = []
    
    seen1 = set()  # This will hold the strings representing the matrices we've seen
    flip = 0
    # Combinations of J unique sequences
    for selected_sequences in itertools.combinations(all_sequences, J):
        Q = np.array(selected_sequences)
        # Canonical form of Q: sorted by columns
        Q_canonical = sort_binary_matrix(Q, columnfirst = flip)
        # Convert to string
        Q_string = binary_matrix_to_string(Q_canonical)
        # If we haven't seen this canonical form before, add it to the 'seen' set and to the list of all matrices
        if Q_string not in seen1:
            seen1.add(Q_string)
            canonical_matrices.append(Q_canonical)
        

    minsize = 2**(J*K)

    while len(canonical_matrices) < minsize:
        minsize = len(canonical_matrices)
        flip = not flip
        seen = set()  # This will hold the strings representing the matrices we've seen
        all_canonical_matrices = canonical_matrices
        canonical_matrices = []
        for Q in all_canonical_matrices:
            # Canonical form of Q: sorted by rows
            Q_canonical = sort_binary_matrix(Q, columnfirst = flip)

            # Convert to string
            Q_string = binary_matrix_to_string(Q_canonical)

            # If we haven't seen this canonical form before, add it to the 'seen' set and to the list of all matrices
            if Q_string not in seen:
                seen.add(Q_string)
                canonical_matrices.append(Q_canonical)
        
        seen = set()
        all_canonical_matrices = canonical_matrices
        canonical_matrices = []
        for Q in all_canonical_matrices:
            # Canonical form of Q: sorted by rows
            Q_canonical = sort_lexicographically(Q, not flip)

            # Convert to string
            Q_string = binary_matrix_to_string(Q_canonical)

            # If we haven't seen this canonical form before, add it to the 'seen' set and to the list of all matrices
            if Q_string not in seen:
                seen.add(Q_string)
                canonical_matrices.append(Q_canonical)
                
    return canonical_matrices



# Example usage:
if __name__ == '__main__':
    # For example, generate all canonical matrices for J=5, K=4.
    canonical_mats = list(generate_canonical_matrices(5, 4))
    print(f"Number of canonical matrices for J=5, K=4: {len(canonical_mats)}")
    
    # Print the first few for inspection.
    for Q in canonical_mats[:5]:
        print(Q)
        print("-----")
