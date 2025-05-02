def add_sequential_counter(row_vars, t, next_var, clauses):
    """
    Sinz sequential-counter encoding of
      sum(row_vars) <= t

    row_vars : list of int
        the DIMACS variable indices for x_{j,1..K}
    t : int
        the bound norm[j]
    next_var : int
        the next unused DIMACS variable index
    clauses : list of list of int
        CNF clauses (to append to)
    returns: new next_var
    """
    K = len(row_vars)
    # If t >= K, no constraint needed
    if t >= K:
        return next_var

    # Create s[i][h] for i=1..K, h=1..(t+1)
    s = [[None]*(t+2) for _ in range(K+1)]
    for i in range(1, K+1):
        for h in range(1, t+2):
            s[i][h] = next_var
            next_var += 1

    # Base: i=1
    #  (¬x1 ∨ s[1][1])
    clauses.append([-row_vars[0], s[1][1]])
    #  for h=2..t+1: ¬s[1][h]  (can't have ≥2...≥t+1 in first var)
    for h in range(2, t+2):
        clauses.append([-s[1][h]])

    # Recurrence for i=2..K
    for i in range(2, K+1):
        xi = row_vars[i-1]
        # (¬xi ∨ s[i][1])
        clauses.append([-xi, s[i][1]])
        # (¬s[i-1][1] ∨ s[i][1])
        clauses.append([-s[i-1][1], s[i][1]])

        # for h=2..t+1:
        for h in range(2, t+2):
            # s[i][h] → s[i-1][h]
            clauses.append([-s[i][h], s[i-1][h]])
            # s[i][h] → xi
            clauses.append([-s[i][h], xi])
            # (s[i-1][h-1] ∧ xi) → s[i][h]
            clauses.append([-s[i-1][h-1], -xi, s[i][h]])

    # Finally forbid ≥ t+1 trues:
    # ¬s[K][t+1]
    clauses.append([-s[K][t+1]])

    return next_var


def generate_row_bounds_cnf(J, K, norm, filename):
    """
    J, K : dimensions
    norm : list of length J, each an int ≤ K
    """
    # x_{j,k} are variables 1..J*K in row-major order
    def var_x(j, k):   
        return j*K + k + 1

    clauses = []
    next_var = J*K + 1

    # add one sequential-counter per row j
    for j in range(J):
        row = [var_x(j, k) for k in range(K)]
        next_var = add_sequential_counter(row, norm[j], next_var, clauses)

    num_vars = next_var - 1
    num_clauses = len(clauses)

    # write DIMACS
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for cl in clauses:
            f.write(" ".join(str(l) for l in cl) + " 0\n")

            
            
def generate_row_bounds_cnf(J, K, norm, filename):
    """
    Enforce for each row j:
      1 <= sum_{k=1}^K x_{j,k} <= norm[j]
    via CNF for Kissat.
    """
    def var_x(j, k):
        return j*K + k + 1

    clauses = []
    next_var = J*K + 1

    for j in range(J):
        row = [var_x(j, k) for k in range(K)]

        # --- 1) Enforce LOWER BOUND: sum(row) >= 1 -------------
        # single OR-clause: at least one x_jk is true
        clauses.append(row.copy())

        # --- 2) Enforce UPPER BOUND via sequential-counter ---
        next_var = add_sequential_counter(row, norm[j], next_var, clauses)

    # write out DIMACS
    num_vars = next_var - 1
    num_clauses = len(clauses)
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for cl in clauses:
            f.write(" ".join(str(l) for l in cl) + " 0\n")
