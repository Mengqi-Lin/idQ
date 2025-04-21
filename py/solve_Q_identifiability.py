import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB


def row_masks(Q: np.ndarray):
    """Return list of integer bit‑masks, one per row of Q."""
    J, K = Q.shape
    # Pack each row into a Python int: bit k = Q[j,k]
    masks = []
    for j in range(J):
        bits = 0
        for k in range(K):
            if Q[j, k]:
                bits |= 1 << k            # set k‑th bit
        masks.append(bits)
    return masks

def classify_row_pairs(Q):
    """
    Partition all unordered pairs (j,j')  (j<j') into three categories:
      * parallel_pairs      : incomparable
      * strict_pairs_fwd    : q_j < q_j'
      * strict_pairs_back   : q_j' < q_j
    Returns three Python lists of tuples (j, j').
    """
    masks = row_masks(Q)
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


def solve_Q_identifiability(Q):
    J, K = Q.shape

    # Step 1: Generate constraints Phi_h based on Q
    def generate_constraints(Q):
        unique_cols = set()
        for a in itertools.product([0, 1], repeat=K):
            a_arr = np.array(a)
            response = np.all(Q <= a_arr, axis=1).astype(int)
            unique_cols.add(tuple(response))
        return list(unique_cols)

    Phi = generate_constraints(Q)
    H = len(Phi)

    # Step 2: Set up the integer program
    model = gp.Model('Q_identifiability')
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")
    a = model.addVars(H, K, vtype=GRB.BINARY, name="a")

    # Feasibility constraints
    for j in range(J):
        for k in range(K):
            model.addConstr(x[j, k] <= Q[j, k])

    # Non-equality constraint to ensure x != Q
    model.addConstr(gp.quicksum(Q[j, k] - x[j, k] for j in range(J) for k in range(K)) >= 1)

    # Logical constraints to define a[h,k]
    for h, phi_h in enumerate(Phi):
        S_h = [j for j in range(J) if phi_h[j] == 1]
        for k in range(K):
            for j in S_h:
                model.addConstr(a[h, k] >= x[j, k])
            model.addConstr(a[h, k] <= gp.quicksum(x[j, k] for j in S_h))

    # Non-domination constraints
    for h, phi_h in enumerate(Phi):
        not_S_h = [j for j in range(J) if phi_h[j] == 0]
        for j in not_S_h:
            model.addConstr(gp.quicksum((1 - a[h, k]) * x[j, k] for k in range(K)) >= 1)

            
    parallel, forward, backward = classify_row_pairs(Q)

    # then generate the inequalities exactly as before, e.g.
    for (j, jp) in parallel:
        model.addConstr(gp.quicksum(x[jp, k] - x[j, k] for k in range(K)) >= 1)
        model.addConstr(gp.quicksum(x[j,  k] - x[jp, k] for k in range(K)) >= 1)

    for (j, jp) in forward:
        model.addConstr(gp.quicksum(x[jp, k] - x[j, k] for k in range(K)) >= 1)

    for (j, jp) in backward:
        model.addConstr(gp.quicksum(x[j,  k] - x[jp, k] for k in range(K)) >= 1)

            
    # Solve the integer program
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
        return solution  # A solution exists
    else:
        return None  # No solution exists

