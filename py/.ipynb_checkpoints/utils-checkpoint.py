import numpy as np
from functools import lru_cache
import itertools
from itertools import product

def unique_pattern_supports(Q):
    """
    Returns all nontrivial supports S = {j | \aaa ⪰ q_j}, for each representative \aaa.
    *excluding* the empty set and the full set {0,…,J-1}.
    """
    J, K = Q.shape
    R = {tuple([0]*K)}
    for j in range(J):
        row = Q[j]
        new = []
        for aaa in R:
            merged = tuple(aaa[k] | row[k] for k in range(K))
            if merged not in R:
                new.append(merged)
        R.update(new)
    # Build and filter supports
    full = frozenset(range(J))
    supports = []
    for aaa in R:
        S = frozenset(j for j in range(J) if all(aaa[k] >= Q[j][k] for k in range(K)))
        if S and S is not full:
            supports.append(set(S))
    return supports



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





def thm_check(Q: np.ndarray, Q_bar: np.ndarray) -> bool:
    """
    Return True  iff  Cols[Φ(Q)] ⊆ Cols[Φ(Q̄)],
    i.e. the non-domination test of your theorem is satisfied.
    """
    J, K = Q.shape
    assert Q_bar.shape == (J, K), "Q and Q_bar must have the same dimensions"

    Phi = unique_response_columns(Q)          # list of 0/1 length-J vectors

    # Pre-compute rows of Q̄ for convenience
    bq = Q_bar.astype(np.uint8)

    # For every φ_a ∈ Φ(Q) ───────────────────────────────────────────────
    for phi_a in Phi:
        S_a      = np.where(phi_a == 1)[0]    # indices with φ_a[j] = 1
        not_S_a  = np.where(phi_a == 0)[0]    # the complement

        # h_a  =  bitwise OR of rows bq_j for j ∈ S_a
        if len(S_a) == 1:
            h_a = bq[S_a[0]].copy()
        else:
            h_a = np.bitwise_or.reduce(bq[S_a], axis=0)

        # Check non-domination for every j ∉ S_a
        # h_a ⪰ bq_j  ⇔  (h_a ≥ bq_j) componentwise
        for j in not_S_a:
            if np.all(h_a >= bq[j]):          # domination ⇒ theorem violated
                return False

    return True




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

