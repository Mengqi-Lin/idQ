import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB
from functools import lru_cache
import itertools
from itertools import product
from ortools.sat.python import cp_model
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Cadical195, Solver

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



def add_lexico_constraints(model, x, J, K):
    """
    Enforce x[:,0] >=_lex x[:,1] >=_lex … >=_lex x[:,K-1]
    via first-difference indicators diff[k,i], i=0..J (i=J means 'no difference').
    
    - model : a Gurobi Model
    - x     : dict or 2D list of x[j,k] binary variables
    - J     : number of rows (items)
    - K     : number of columns (attributes)
    """
    # diff[k,i] = 1 if the *first* row where col k and k+1 differ is row i.
    # i in [0..J-1] covers each item; i=J means 'they never differ' (fully equal).
    diff = model.addVars(K-1, J+1, vtype=GRB.BINARY, name="diff")
    
    for k in range(K-1):
        # exactly one first-difference position
        model.addConstr(gp.quicksum(diff[k,i] for i in range(J+1)) == 1,
                        name=f"lex_sumdiff_{k}")
        
        # if they first differ at row i < J:
        for i in range(J):
            # all rows t < i must be equal:
            for t in range(i):
                model.addConstr(x[t,k] - x[t,k+1] <= 1 - diff[k,i],
                                name=f"lex_eq_above_{k}_{i}_{t}")
                model.addConstr(x[t,k+1] - x[t,k] <= 1 - diff[k,i],
                                name=f"lex_eq_above_rev_{k}_{i}_{t}")
            # at row i, enforce x[i,k]=1 and x[i,k+1]=0
            model.addConstr(x[i,k]     >= diff[k,i],
                            name=f"lex_diff1_{k}_{i}")
            model.addConstr(x[i,k+1] <= 1 - diff[k,i],
                            name=f"lex_diff2_{k}_{i}")
        
        # if no difference (i = J), all rows must be equal
        for t in range(J):
            model.addConstr(x[t,k] - x[t,k+1] <= 1 - diff[k,J],
                            name=f"lex_eq_nodiff_{k}_{t}")
            model.addConstr(x[t,k+1] - x[t,k] <= 1 - diff[k,J],
                            name=f"lex_eq_nodiff_rev_{k}_{t}")


def solve_Q_identifiability_IPfast(Q):
    J, K = Q.shape

    # Step 1: Generate constraints Phi_a based on Q
    supports = unique_pattern_supports(Q)
    A = len(supports)


    # Step 2: Set up the integer program
    model = gp.Model('Q_identifiability')
    model.setParam('SolutionLimit', 1)
    model.setParam('OutputFlag', 1)

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
    for a, S_a in enumerate(supports):
        for k in range(K):
            for j in S_a:
                model.addConstr(h[a, k] >= x[j, k], name=f"h_lb_a{a}_j{j}_k{k}")
            model.addConstr(h[a, k] <= gp.quicksum(x[j, k] for j in S_a), name=f"h_ub_a{a}_k{k}")

    # Non-domination constraint: h(a) ≰ \bqq_j for all j not in S_a
    for a, S_a in enumerate(supports):
        for j in range(J):
            if j not in S_a:
                model.addConstr(
                    gp.quicksum((1 - h[a, k]) * x[j, k] for k in range(K)) >= 1,
                    name=f"non_dom_a{a}_j{j}"
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

    
    
    
def solve_Q_identifiability_IP(Q):
    J, K = Q.shape

    # Step 1: Generate constraints Phi_a based on Q
    supports = unique_pattern_supports(Q)
    A = len(supports)

    # Step 2: Set up the integer program
    model = gp.Model('Q_identifiability')
    model.setParam('SolutionLimit', 1)
    model.setParam('OutputFlag', 1)

    # Decision variables
    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")     # for \bQQ
    h = model.addVars(A, K, vtype=GRB.BINARY, name="h")     # for h(a)

    # Norm constraints, together with the column ordering constraint. Slower than the unproved version "\bqq_j \preceq \qq_j".
        
    # --- norm constraints ------------------------------
    dist = distances(Q)
    for j in range(J):
        model.addConstr(gp.quicksum(x[j, k] for k in range(K)) <= K - dist[j])
        model.addConstr(1 <= gp.quicksum(x[j, k] for k in range(K)))
        
    # --- column lex‑ordering constraints ------------------------------
    add_lexico_constraints(model, x, J, K) 
        
    # Non-equality constraint to ensure \bQQ ≠ \QQ
    model.addConstr(
        gp.quicksum(
            (1 - Q[j, k]) * x[j, k]         # q=0  →  uses x
            + Q[j, k]    * (1 - x[j, k])    # q=1  →  uses 1-x
            for j in range(J)
            for k in range(K)
        ) >= 1,name="Q != Q_bar")

    # Constraints to define h[a,k] = ⋁_{j in S_a} x[j,k]
    for a, S_a in enumerate(supports):
        for k in range(K):
            for j in S_a:
                model.addConstr(h[a, k] >= x[j, k], name=f"h_lb_a{a}_j{j}_k{k}")
            model.addConstr(h[a, k] <= gp.quicksum(x[j, k] for j in S_a), name=f"h_ub_a{a}_k{k}")

    # Non-domination constraint: h(a) ≰ \bqq_j for all j not in S_a
    for a, S_a in enumerate(supports):
        for j in range(J):
            if j not in S_a:
                model.addConstr(
                    gp.quicksum((1 - h[a, k]) * x[j, k] for k in range(K)) >= 1,
                    name=f"non_dom_a{a}_j{j}"
                )


    # Pairwise constraints (each pair of columns must have both (0,1) and (1,0))
    for p, q in itertools.combinations(range(K), 2):
        model.addConstr(gp.quicksum(x[j, p] * (1 - x[j, q]) for j in range(J)) >= 1)
        model.addConstr(gp.quicksum((1 - x[j, p]) * x[j, q] for j in range(J)) >= 1)
        
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

    

# def solve_identifiability_Q_cpsat(Q):
#     """
#     CP-SAT feasibility check for Q-matrix identifiability.

#     Arguments:
#       Q:        A J×K binary matrix (list of lists or numpy array).

#     Returns:
#       (identifiable, Qbar_candidate)
#       - identifiable = True  if no alternate Qbar exists (so Q IS identifiable),
#                        False if we find a Qbar ≉ Q that explains all patterns.
#       - Qbar = the found J×K matrix (list of lists) when identifiable=False,
#                          otherwise None.
#     """
#     J, K = Q.shape

#     # 1) Compute unique response patterns Phi
#     supports = unique_pattern_supports(Q)
#     A = len(supports)

#     model = cp_model.CpModel()

#     # 2) Decision variables
#     x = [[model.NewBoolVar(f"x_{j}_{k}") for k in range(K)] for j in range(J)]
#     h = [[model.NewBoolVar(f"h_{a}_{k}") for k in range(K)] for a in range(A)]

#     # --- norm constraints ------------------------------
#     # compute dist[j] = d(q_j) from your existing distances(Q) function
#     dist = distances(Q)  # array of length J

#     for j in range(J):
#         # at least one skill per item
#         model.Add( sum(x[j][k] for k in range(K)) >= 1 )
#         # at most K - dist[j] skills per item
#         model.Add( sum(x[j][k] for k in range(K)) <= K - dist[j] )

    
#     # 3) OR-constraint: for each pattern a and each skill k,
#     #    h[a][k] == OR_{j in S_a} x[j][k],  where S_a = { j | Phi[a][j] == 1 }
#     for a, S_a in enumerate(supports):
#         for k in range(K):
#             if S_a:
#                 # If any x[j,k] is 1 for j∈S, then h[a][k] must be 1:
#                 for j in S_a:
#                     model.AddImplication(x[j][k], h[a][k])
#                 # If all x[j,k]==0 for j∈S, then h[a][k] must be 0:
#                 #   (¬x[j1,k] ∧ ¬x[j2,k] ∧ …) ⇒ ¬h[a][k]
#                 clause = [x[j][k].Not() for j in S_a] + [h[a][k].Not()]
#                 model.AddBoolOr(clause)
#             else:
#                 # If pattern a has no correct items at all, it can never have any skill:
#                 model.Add(h[a][k] == 0)

#     # 4) Non-covering constraint: for each pattern a and each item j NOT in S_a,
#     #    ∃k s.t. x[j][k]==1 ∧ h[a][k]==0
#     for a, S_a in enumerate(supports):
#         for j in range(J):
#             if j not in S_a:
#                 # Build a small auxiliary b_{a,j,k} for each k:
#                 b_vars = []
#                 for k in range(K):
#                     b = model.NewBoolVar(f"b_{a}_{j}_{k}")
#                     b_vars.append(b)
#                     # b=1 ⇒ x[j,k]=1
#                     model.AddImplication(b, x[j][k])
#                     # b=1 ⇒ h[a,k]=0
#                     model.AddImplication(b, h[a][k].Not())
#                 # Now require at least one b[a,j,k] = 1
#                 model.AddBoolOr(b_vars)

#     # 5) Forbid the trivial solution x==Q: enforce at least one entry differs
#     diff_clause = []
#     for j in range(J):
#         for k in range(K):
#             if Q[j][k] == 1:
#                 diff_clause.append(x[j][k].Not())
#             else:
#                 diff_clause.append(x[j][k])
#     # This one big OR says "there is at least one (j,k) where x[j,k] ≠ Q[j][k]"
#     model.AddBoolOr(diff_clause)

#     # 6) Lexicographic non-increasing ordering on columns of x:
#     #    For each adjacent pair k, k+1, introduce diff[k][i] for i=0..J
#     #    i in [0..J-1] means first difference at row i; i=J means no difference.
#     for k in range(K-1):
#         diff = [model.NewBoolVar(f"diff_{k}_{i}") for i in range(J+1)]
#         # exactly one position
#         model.Add(sum(diff) == 1)
#         # if first diff at row i < J:
#         for i in range(J):
#             # equality above i
#             for t in range(i):
#                 model.Add(x[t][k] == x[t][k+1]).OnlyEnforceIf(diff[i])
#             # at i: x[i,k]=1, x[i,k+1]=0
#             model.Add(x[i][k] == 1).OnlyEnforceIf(diff[i])
#             model.Add(x[i][k+1] == 0).OnlyEnforceIf(diff[i])
#         # if no difference at all (diff[J]):
#         for t in range(J):
#             model.Add(x[t][k] == x[t][k+1]).OnlyEnforceIf(diff[J])
            
            
#     # 7) Solve
#     solver = cp_model.CpSolver()
#     status = solver.Solve(model)

#     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         # Found an alternate Qbar ⇒ original Q is NOT identifiable
#         Qbar = [[solver.Value(x[j][k]) for k in range(K)] for j in range(J)]
#         return Qbar
#     else:
#         # No alternate found within time ⇒ Q is identifiable
#         return None

    
    
def solve_identifiability_Q_cpsat(Q):
    model = cp_model.CpModel()
    J = len(Q)             # number of items (rows)
    K = len(Q[0]) if J > 0 else 0  # number of attributes (columns)
    # Compute dist[j] as the number of 1s in row j of original Q (assuming this interpretation)
    dist = distances(Q)

    # Get unique response patterns of Q. S_patterns[a] is the set of item indices that pattern a solved.
    S_patterns = unique_pattern_supports(Q)  # assume this function is provided
    A = len(S_patterns)  # number of unique response patterns

    # Create decision variables x[j][k] and h[a][k]
    x = [[model.NewBoolVar(f"x[{j},{k}]") for k in range(K)] for j in range(J)]
    h = [[model.NewBoolVar(f"h[{a},{k}]") for k in range(K)] for a in range(A)]

    # Norm constraints: 1 <= sum_k x[j,k] <= K - dist[j] for each item j
    for j in range(J):
        # At least one attribute per item
        model.Add(sum(x[j][k] for k in range(K)) >= 1)
        # At most K - dist[j] attributes for item j
        model.Add(sum(x[j][k] for k in range(K)) <= K - dist[j])

    # OR constraints: h[a,k] = OR_{j in S_a} x[j,k]
    for a, S_a in enumerate(S_patterns):
        for k in range(K):
            # If any j in S_a requires k, then h[a,k] must be 1.
            for j in S_a:
                model.Add(x[j][k] <= h[a][k])
            # If h[a,k] is 1, at least one j in S_a has x[j,k] = 1.
            # (If S_a is empty, pattern a solved no items, then h[a,k] should be 0 for all k.)
            if len(S_a) > 0:
                model.Add(sum(x[j][k] for j in S_a) >= 1).OnlyEnforceIf(h[a][k])
            else:
                model.Add(h[a][k] == 0)

    # Non-domination constraints: for each pattern a and item j not in S_a, 
    # there exists at least one k with x[j,k]=1 and h[a,k]=0.
    # We introduce y_{j,a,k} variables to capture (x[j,k] AND NOT h[a,k]).
    y = {}  # dictionary for y[(j,a,k)] variables
    for a, S_a in enumerate(S_patterns):
        for j in range(J):
            if j in S_a:
                continue  # skip items that pattern a already solves
            for k in range(K):
                y[(j,a,k)] = model.NewBoolVar(f"y[{j},{a},{k}]")
                # y[j,a,k] -> x[j,k] and y[j,a,k] -> (NOT h[a,k])
                model.Add(x[j][k] == 1).OnlyEnforceIf(y[(j,a,k)])
                model.Add(h[a][k] == 0).OnlyEnforceIf(y[(j,a,k)])
                # (x[j,k] and NOT h[a,k]) -> y[j,a,k]
                model.AddBoolAnd([x[j][k], h[a][k].Not()]).OnlyEnforceIf(y[(j,a,k)])
                # If x[j,k] and h[a,k] are such that x=1 and h=0, then y can be 1.
                # (Note: AddBoolAnd with OnlyEnforceIf acts as implication: if y is true, then x=1 and h=0.
                # We also need the reverse, which we handle by the OnlyEnforceIf above.)
            # At least one y[j,a,k] must be true (>=1) for j not in S_a
            # (i.e., item j has some required attr missing in pattern a)
            model.Add(sum(y[(j,a,k)] for k in range(K)) >= 1)

    # Lexicographic ordering constraints: ensure columns of x are sorted non-increasing lexicographically.
    for k in range(K - 1):
        # prefix_equal[i] is true iff x[0..i-1, k] == x[0..i-1, k+1].
        prefix_equal = [model.NewBoolVar(f"prefix_equal_{k}_{i}") 
                        for i in range(J+1)]
        model.Add(prefix_equal[0] == 1)  # empty prefix is equal

        for i in range(J):
            # eq = True iff x[i,k] == x[i,k+1].
            eq = model.NewBoolVar(f"eq_{k}_{i}")
            # Enforce eq ↔ (x[i,k] == x[i,k+1]):
            model.Add(x[i][k] == x[i][k+1]).OnlyEnforceIf(eq)
            model.Add(x[i][k] + x[i][k+1] == 1).OnlyEnforceIf(eq.Not())

            # Compute prefix_equal[i+1] = prefix_equal[i] AND eq
            # (Using linear reification for AND)
            model.Add(prefix_equal[i+1] <= prefix_equal[i])
            model.Add(prefix_equal[i+1] <= eq)
            model.Add(prefix_equal[i+1] >= prefix_equal[i] + eq - 1)

            # If prefix_equal[i] is true, enforce the lex-order at row i
            # (i.e. x[i][k] >= x[i][k+1])
            model.Add(x[i][k] >= x[i][k+1]).OnlyEnforceIf(prefix_equal[i])
 
    # Non-equality constraint: ensure x (the new Q) differs from original Q in at least one position.
    diff_literals = []
    for j in range(J):
        for k in range(K):
            if Q[j][k] == 1:
                # if original is 1, then requiring a difference means x[j][k] must be 0 for this clause to satisfy.
                diff_literals.append(x[j][k].Not())
            else:
                # if original is 0, difference means x[j][k] = 1 for that entry.
                diff_literals.append(x[j][k])
    model.AddBoolOr(diff_literals)

    # Solve the CP-SAT model
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True

    result_status = solver.Solve(model)
    if result_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Extract solution for x
        solution = [[int(solver.Value(x[j][k])) for k in range(K)] for j in range(J)]
        return solution
    else:
        return None  # No alternative Q found (Q might be identifiable)

    
    
    
    


def solve_SAT(Q, Cardbound=None, solver_name='cadical195'):
    Q = np.asarray(Q, dtype=int)
    J, K = Q.shape
    # default Cardbound is the original row‐sums
    if Cardbound is None:
        Cardbound = Q.sum(axis=1).tolist()

    pool = IDPool()
    cnf   = CNF()

    # --- 1) build X as a J×K matrix of fresh vars: X[j][k] = x_{j,k} ---
    X = [[ pool.id(('x', j, k)) for k in range(K)] for j in range(J)]

    # --- 2) lex‐decreasing columns:  
        # for each k: compare column k vs k+1
    for k in range(K - 1):
        col_k   = [X[j][k]   for j in range(J)]
        col_k1  = [X[j][k+1] for j in range(J)]
        add_lex_gt(cnf, col_k, col_k1, pool)

    # --- 3) row‐bounds: 1 ≤ sum_k X[j][k] ≤ Cardbound[j] ---
    for j in range(J):
        row_lits = X[j]
        # at most
        cnf.extend(
            CardEnc.atmost(lits=row_lits,
                           bound=Cardbound[j],
                           encoding=EncType.seqcounter).clauses
        )
        # at least one
        cnf.append(row_lits)

    # --- 4) X ≠ Q: one entry must differ ---
    diff_clause = []
    for j in range(J):
        for k in range(K):
            lit = X[j][k] if Q[j, k] == 0 else -X[j][k]
            diff_clause.append(lit)
    cnf.append(diff_clause)

    # --- 5) solve ---
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as s:
        if not s.solve():
            return None

        model = set(s.get_model())
        X_mat = np.zeros_like(Q)
        for j in range(J):
            for k in range(K):
                if X[j][k] in model:
                    X_mat[j, k] = 1
        return X_mat
