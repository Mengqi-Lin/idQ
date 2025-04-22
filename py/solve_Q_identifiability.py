import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB
from functools import lru_cache

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



def classify_row_pairs(Q):
    """
    Partition all unordered pairs (j,j')  (j<j') into three categories:
      * parallel_pairs      : incomparable
      * strict_pairs_fwd    : q_j < q_j'
      * strict_pairs_back   : q_j' < q_j
    Returns three Python lists of tuples (j, j').
    """
    masks, _ = row_masks(Q)
    J = len(masks)

    parallel_pairs      = []
    strict_pairs_fwd    = []
    strict_pairs_back   = []

    for j in range(J):
        mj = masks[j]
        for jp in range(j + 1, J):
            mjp = masks[jp]

            #  q_j <= q_j'  ⇔  (mj | mjp) == mjp
            le = (mj | mjp) == mjp
            ge = (mj | mjp) == mj

            if le and not ge:            #  q_j < q_j'
                strict_pairs_fwd.append((j, jp))
            elif ge and not le:          #  q_j' < q_j
                strict_pairs_back.append((j, jp))
            else:                        #  incomparable
                parallel_pairs.append((j, jp))

    return parallel_pairs, strict_pairs_fwd, strict_pairs_back



def solve_Q_identifiability_fast(Q):
    J, K = Q.shape

    # Step 1: Generate constraints Phi_a based on Q
    Phi = list(unique_response_columns(Q))
    A = len(Phi)  # number of unique response vectors

    # Step 2: Set up the integer program
    model = gp.Model('Q_identifiability')
    model.setParam('SolutionLimit', 1)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")     # for \bQQ
    h = model.addVars(A, K, vtype=GRB.BINARY, name="h")     # for h(a)

    # Feasibility constraints (Unproved: \bqq_j \preceq \qq_j)
    for j in range(J):
        for k in range(K):
            model.addConstr(x[j, k] <= Q[j, k], name=f"\bq_{j}_{k} <= q_{j}_{k}")

    # Non-equality constraint to ensure \bQQ ≠ \QQ
    model.addConstr(
        gp.quicksum(Q[j, k] - x[j, k] for j in range(J) for k in range(K)) >= 1,
        name="Q != Q_bar"
    )

    # Constraints to define h[a,k] = ⋁_{j in S_a} x[j,k]
    for a, phi_a in enumerate(Phi):
        S_a = [j for j in range(J) if phi_a[j] == 1]
        for k in range(K):
            for j in S_a:
                model.addConstr(h[a, k] >= x[j, k], name=f"h_lb_a{a}_j{j}_k{k}")
            model.addConstr(h[a, k] <= gp.quicksum(x[j, k] for j in S_a), name=f"h_ub_a{a}_k{k}")

    # Non-domination constraint: h(a) ≰ \bqq_j for all j not in S_a
    for a, phi_a in enumerate(Phi):
        not_S_a = [j for j in range(J) if phi_a[j] == 0]
        for j in not_S_a:
            model.addConstr(
                gp.quicksum((1 - h[a, k]) * x[j, k] for k in range(K)) >= 1,
                name=f"non_dom_a{a}_j{j}"
            )

    # Partial order constraints between rows
    parallel, forward, backward = classify_row_pairs(Q)

    for (j, jp) in parallel:
        # bq_j  not <=  bq_jp
        model.addConstr(
            gp.quicksum((1 - x[jp, k]) * x[j, k] for k in range(K)) >= 1,
            name=f"parallel_{j}_{jp}_a"
        )
        # bq_jp not <=  bq_j
        model.addConstr(
            gp.quicksum((1 - x[j,  k]) * x[jp, k] for k in range(K)) >= 1,
            name=f"parallel_{j}_{jp}_b"
        )

    for (j, jp) in forward:   #  qq_j < qq_jp  ⇒  forbid  bq_jp <= bq_j
        model.addConstr(
            gp.quicksum((1 - x[j, k]) * x[jp, k] for k in range(K)) >= 1,
            name=f"forward_{j}_{jp}"
        )

    for (j, jp) in backward:  #  qq_jp < qq_j  ⇒  forbid  bq_j  <= bq_jp
        model.addConstr(
            gp.quicksum((1 - x[jp, k]) * x[j, k] for k in range(K)) >= 1,
            name=f"backward_{j}_{jp}"
        )

    # Solve the integer program
    model.optimize()
    status = model.Status

    if status == GRB.INFEASIBLE:
        return None                         # Q identifiable
    elif status in (GRB.OPTIMAL, GRB.SOLUTION_LIMIT):
        solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
        return solution  # Q is not identifiable
    elif status == GRB.SUBOPTIMAL:
        if model.SolCount > 0:
            solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
            return solution
        else:
            return None  # inconclusive — could trigger Phase B
    else:
        return None      # early termination (time limit, etc.)

    
    
    
def solve_Q_identifiability(Q):
    J, K = Q.shape

    # Step 1: Generate constraints Phi_a based on Q
    Phi = list(unique_response_columns(Q))
    A = len(Phi)  # number of unique response vectors

    # Step 2: Set up the integer program
    model = gp.Model('Q_identifiability')
    model.setParam('SolutionLimit', 1)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")     # for \bQQ
    h = model.addVars(A, K, vtype=GRB.BINARY, name="h")     # for h(a)

    # Norm constraints, together with the column ordering constraint. Slower than the unproved version "\bqq_j \preceq \qq_j".
    
    # --- norm constraints ------------------------------
    dist = distances(Q)
    for j in range(J):
        model.addConstr(gp.quicksum(x[j, k] for k in range(K)) <= K - dist[j])
        
    # --- column lex‑ordering constraints ------------------------------
    pow2 = [1 << (J - 1 - j) for j in range(J)]   # 2^{J-j-1}
    def column_code(k):
        return gp.quicksum(pow2[j] * x[j, k] for j in range(J))
    for k in range(K - 1):
        model.addConstr(column_code(k) >= column_code(k + 1),
                        name=f"lex_col_{k}") 
        
        
    # Non-equality constraint to ensure \bQQ ≠ \QQ
    model.addConstr(
        gp.quicksum(
            (1 - Q[j, k]) * x[j, k]         # q=0  →  uses x
            + Q[j, k]    * (1 - x[j, k])    # q=1  →  uses 1-x
            for j in range(J)
            for k in range(K)
        ) >= 1,name="Q != Q_bar")

    # Constraints to define h[a,k] = ⋁_{j in S_a} x[j,k]
    for a, phi_a in enumerate(Phi):
        S_a = [j for j in range(J) if phi_a[j] == 1]
        for k in range(K):
            for j in S_a:
                model.addConstr(h[a, k] >= x[j, k], name=f"h_lb_a{a}_j{j}_k{k}")
            model.addConstr(h[a, k] <= gp.quicksum(x[j, k] for j in S_a), name=f"h_ub_a{a}_k{k}")

    # Non-domination constraint: h(a) ≰ \bqq_j for all j not in S_a
    for a, phi_a in enumerate(Phi):
        not_S_a = [j for j in range(J) if phi_a[j] == 0]
        for j in not_S_a:
            model.addConstr(
                gp.quicksum((1 - h[a, k]) * x[j, k] for k in range(K)) >= 1,
                name=f"non_dom_a{a}_j{j}"
            )

    # Partial order constraints between rows
    parallel, forward, backward = classify_row_pairs(Q)

        
    # ------------------------------------------------------------------
    # product‑sum non‑domination constraints
    # ------------------------------------------------------------------
    for (j, jp) in parallel:
        # bq_j  not <=  bq_jp
        model.addConstr(
            gp.quicksum((1 - x[jp, k]) * x[j, k] for k in range(K)) >= 1,
            name=f"parallel_{j}_{jp}_a"
        )
        # bq_jp not <=  bq_j
        model.addConstr(
            gp.quicksum((1 - x[j,  k]) * x[jp, k] for k in range(K)) >= 1,
            name=f"parallel_{j}_{jp}_b"
        )

    for (j, jp) in forward:   #  qq_j < qq_jp  ⇒  forbid  bq_jp <= bq_j
        model.addConstr(
            gp.quicksum((1 - x[j, k]) * x[jp, k] for k in range(K)) >= 1,
            name=f"forward_{j}_{jp}"
        )

    for (j, jp) in backward:  #  qq_jp < qq_j  ⇒  forbid  bq_j  <= bq_jp
        model.addConstr(
            gp.quicksum((1 - x[jp, k]) * x[j, k] for k in range(K)) >= 1,
            name=f"backward_{j}_{jp}"
        )

        
    # Solve the integer program
    model.optimize()
    status = model.Status

    if status == GRB.INFEASIBLE:
        return None                         # Q identifiable
    elif status in (GRB.OPTIMAL, GRB.SOLUTION_LIMIT):
        solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
        return solution  # Q is not identifiable
    elif status == GRB.SUBOPTIMAL:
        if model.SolCount > 0:
            solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
            return solution
        else:
            return None  # inconclusive — could trigger Phase B
    else:
        return None      # early termination (time limit, etc.)
