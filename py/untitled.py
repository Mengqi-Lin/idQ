    from itertools import combinations
    for (a, b, c) in combinations(range(K), 3):
        # Introduce an auxiliary var for "this triple has a pure item"
        triple_ok = model.addVar(vtype=gp.GRB.BINARY, name=f"triple_{a}_{b}_{c}")
        # At least one item satisfies one of the pure conditions for this triple -> triple_ok = 1
        model.addConstr(
            gp.quicksum(
                (Q_vars[j][a]) * (1 - Q_vars[j][b]) * (1 - Q_vars[j][c]) +
                (Q_vars[j][b]) * (1 - Q_vars[j][a]) * (1 - Q_vars[j][c]) +
                (Q_vars[j][c]) * (1 - Q_vars[j][a]) * (1 - Q_vars[j][b])
                for j in range(J)
            ) >= triple_ok, name=f"exists_pure_{a}_{b}_{c}"
        )