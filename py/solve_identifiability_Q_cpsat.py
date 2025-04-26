from ortools.sat.python import cp_model

def solve_identifiability_Q_cpsat(Q, time_limit=30):
    """
    CP-SAT feasibility check for Q-matrix identifiability.

    Arguments:
      Q:        A J×K binary matrix (list of lists or numpy array).
      time_limit: maximum solve time in seconds (default 30).

    Returns:
      (identifiable, Qbar_candidate)
      - identifiable = True  if no alternate Qbar exists (so Q IS identifiable),
                       False if we find a Qbar ≉ Q that explains all patterns.
      - Qbar_candidate = the found J×K matrix (list of lists) when identifiable=False,
                         otherwise None.
    """
    J, K = len(Q), len(Q[0])

    # 1) Compute unique response patterns Phi
    # Each pattern is length-J tuple of 0/1’s: 1 means "correct" under that latent a.
    from itertools import product
    responses = set()
    for alpha in product([0,1], repeat=K):
        resp = []
        for j in range(J):
            # item j requires Q[j][k]==1 for all k in alpha
            ok = 1
            for k in range(K):
                if Q[j][k] == 1 and alpha[k] == 0:
                    ok = 0
                    break
            resp.append(ok)
        responses.add(tuple(resp))
    Phi = list(responses)
    A = len(Phi)

    model = cp_model.CpModel()

    # 2) Decision variables
    x = [[model.NewBoolVar(f"x_{j}_{k}") for k in range(K)] for j in range(J)]
    h = [[model.NewBoolVar(f"h_{a}_{k}") for k in range(K)] for a in range(A)]

    # 3) OR-constraint: for each pattern a and each skill k,
    #    h[a][k] == OR_{j in S_a} x[j][k],  where S_a = { j | Phi[a][j] == 1 }
    for a, pattern in enumerate(Phi):
        S = [j for j,resp in enumerate(pattern) if resp == 1]
        for k in range(K):
            if S:
                # If any x[j,k] is 1 for j∈S, then h[a][k] must be 1:
                for j in S:
                    model.AddImplication(x[j][k], h[a][k])
                # If all x[j,k]==0 for j∈S, then h[a][k] must be 0:
                #   (¬x[j1,k] ∧ ¬x[j2,k] ∧ …) ⇒ ¬h[a][k]
                clause = [x[j][k].Not() for j in S] + [h[a][k].Not()]
                model.AddBoolOr(clause)
            else:
                # If pattern a has no correct items at all, it can never have any skill:
                model.Add(h[a][k] == 0)

    # 4) Non-covering constraint: for each pattern a and each item j NOT in S_a,
    #    ∃k s.t. x[j][k]==1 ∧ h[a][k]==0
    for a, pattern in enumerate(Phi):
        S = set(j for j,resp in enumerate(pattern) if resp == 1)
        for j in range(J):
            if j not in S:
                # Build a small auxiliary b_{a,j,k} for each k:
                b_vars = []
                for k in range(K):
                    b = model.NewBoolVar(f"b_{a}_{j}_{k}")
                    b_vars.append(b)
                    # b=1 ⇒ x[j,k]=1
                    model.AddImplication(b, x[j][k])
                    # b=1 ⇒ h[a,k]=0
                    model.AddImplication(b, h[a][k].Not())
                # Now require at least one b[a,j,k] = 1
                model.AddBoolOr(b_vars)

    # 5) Forbid the trivial solution x==Q: enforce at least one entry differs
    diff_clause = []
    for j in range(J):
        for k in range(K):
            if Q[j][k] == 1:
                diff_clause.append(x[j][k].Not())
            else:
                diff_clause.append(x[j][k])
    # This one big OR says "there is at least one (j,k) where x[j,k] ≠ Q[j][k]"
    model.AddBoolOr(diff_clause)

    # 6) Lexicographic non-increasing ordering on columns of x:
    #    For each adjacent pair k, k+1, introduce diff[k][i] for i=0..J
    #    i in [0..J-1] means first difference at row i; i=J means no difference.
    for k in range(K-1):
        diff = [model.NewBoolVar(f"diff_{k}_{i}") for i in range(J+1)]
        # exactly one position
        model.Add(sum(diff) == 1)
        # if first diff at row i < J:
        for i in range(J):
            # equality above i
            for t in range(i):
                model.Add(x[t][k] == x[t][k+1]).OnlyEnforceIf(diff[i])
            # at i: x[i,k]=1, x[i,k+1]=0
            model.Add(x[i][k] == 1).OnlyEnforceIf(diff[i])
            model.Add(x[i][k+1] == 0).OnlyEnforceIf(diff[i])
        # if no difference at all (diff[J]):
        for t in range(J):
            model.Add(x[t][k] == x[t][k+1]).OnlyEnforceIf(diff[J])
            
            
    # 7) Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Found an alternate Qbar ⇒ original Q is NOT identifiable
        Qbar = [[solver.Value(x[j][k]) for k in range(K)] for j in range(J)]
        return False, Qbar
    else:
        # No alternate found within time ⇒ Q is identifiable
        return True, None
