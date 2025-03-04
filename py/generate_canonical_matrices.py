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

def generate_canonical_matrices(J, K):
    """
    Generate all binary matrices of shape (J, K) up to column permutation equivalence.
    Each matrix is returned in its canonical form (i.e. its columns are sorted lexicographically).
    
    Parameters:
        J (int): Number of rows.
        K (int): Number of columns.
        
    Yields:
        np.ndarray: A binary matrix of shape (J, K) in canonical form.
    """
    total = 2 ** (J * K)
    print(f"Enumerating {total} matrices (may take some time for larger J,K)...")
    for bits in itertools.product([0, 1], repeat=J*K):
        Q = np.array(bits).reshape((J, K))
        # Get the canonical form of Q.
        can_Q = canonicalize(Q)
        # Yield Q only if it is already in canonical form.
        if np.array_equal(Q, can_Q):
            yield Q

# Example usage:
if __name__ == '__main__':
    # For example, generate all canonical matrices for J=5, K=4.
    canonical_mats = list(generate_canonical_matrices(5, 4))
    print(f"Number of canonical matrices for J=5, K=4: {len(canonical_mats)}")
    
    # Print the first few for inspection.
    for Q in canonical_mats[:5]:
        print(Q)
        print("-----")
