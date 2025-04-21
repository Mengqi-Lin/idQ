import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB

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

    # Solve the integer program
    model.optimize()

    # Return results
    if model.status == GRB.OPTIMAL:
        solution = np.array([[int(x[j, k].X) for k in range(K)] for j in range(J)])
        return solution  # A solution exists
    else:
        return None  # No solution exists

