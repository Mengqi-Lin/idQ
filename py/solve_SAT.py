import numpy as np
import itertools
from itertools import product
from ortools.sat.python import cp_model
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Cadical195, Solver
from pysat.engines  import BooleanEngine
import time
from utils import unique_pattern_supports, minimal_size_parent, distances


def generate_clause(X):
    clause = []
    for x in X:
        clause.append(x)
    return clause

def generate_implication_clause(X, Y):
    clause = []
    for x in X:
        clause.append(-x)
    for y in Y:
        clause.append(y)
    return clause


def generate_lex_clauses(X, Y, strict, total_vars):
    
    """
    Harvey's linear encoding  (X >_lex Y   if strict else  X ≥_lex Y).

    Returns (clauses, new_total_vars).
    """
    clauses = []
    n = len(X)
    # First bit: X[0] ≥ Y[0]
    clauses.append(generate_implication_clause({Y[0]}, {X[0]}))
    clauses.append(generate_implication_clause({Y[0]}, {total_vars+1}))
    clauses.append(generate_clause({X[0], total_vars+1}))
    
    # For remaining bits: maintain the lexicographic ordering
    for k in range(1, n-1): 
        clauses.append(generate_implication_clause({total_vars+k}, {-Y[k], X[k]}))
        clauses.append(generate_implication_clause({total_vars+k}, {-Y[k], total_vars+k+1}))
        clauses.append(generate_implication_clause({total_vars+k}, {X[k], total_vars+k+1}))
    
    # Handle the last bit
    if strict:
        clauses.append(generate_implication_clause({total_vars+n-1}, {-Y[n-1]}))
        clauses.append(generate_implication_clause({total_vars+n-1}, {X[n-1]}))
    else:
        clauses.append(generate_implication_clause({total_vars+n-1}, {-Y[n-1], X[n-1]}))
    return (clauses, total_vars+n-1)



def add_lex_decreasing(cnf, pool, X, J, K):
    """
    Append clauses forcing columns of X to be *strictly* decreasing
    in lexicographic order:  X[:,0] >_lex X[:,1] >_lex … > X[:,K-1].

    Uses the generate_lex_clauses() helper above.
    """
    total_vars = pool.top  # current highest variable ID

    for k in range(K - 1):
        col_k  = [X[j][k]   for j in range(J)]
        col_k1 = [X[j][k+1] for j in range(J)]

        clauses, total_vars = generate_lex_clauses(
            col_k, col_k1, strict=True, total_vars=total_vars
        )
        cnf.extend(clauses)

    # tell the pool about the new auxiliary variables
    pool.top = total_vars


# ----------------------------------------------------------------------
#  Constraint:  X != Q
# ----------------------------------------------------------------------
def add_neq_Q(cnf, X, Q):
    diff_clause = []
    for j, row in enumerate(Q):
        for k, qjk in enumerate(row):
            diff_clause.append(-X[j][k] if qjk else X[j][k])
    cnf.append(diff_clause)

# ----------------------------------------------------------------------
#  Constraint:  X <= Q
# ----------------------------------------------------------------------

def add_leq_Q(cnf, X, Q):
    for j, row in enumerate(Q):
        for k, qjk in enumerate(row):
            if qjk == 0:
                cnf.append([-X[j][k]])  # Enforce x[j][k] == 0


# ----------------------------------------------------------------------
#  Constraint:  1 <= row-sum <= Cardbound[j]
# ----------------------------------------------------------------------
def add_row_cardinality(cnf, pool, X, Cardbound, encoding=EncType.seqcounter):
    J, K = len(X), len(X[0])
    for j in range(J):
        row_lits = X[j]
        # at least one 1
        cnf.append(row_lits)
        # at most Cardbound[j] ones
        b = Cardbound[j]
        enc = CardEnc.atmost(lits=row_lits, bound=b, vpool=pool, encoding=encoding)
        cnf.extend(enc.clauses)


# ----------------------------------------------------------------------
#  Constraint:  exact U-constraint
# ----------------------------------------------------------------------

def add_U_constraint(cnf, pool, X, U, parent):
    """
    Add only the necessary C2–witness clauses, skipping maximal supports.

    Parameters
    ----------
    cnf    : list[list[int]]
    pool   : IDPool
    X      : list[list[int]]       # X[j][k] is literal for x_{j,k}
    U      : list[list[int]]       # Supports S_Q(\aaa) for all \aaa\in R.
    parent : list[int | None]      # one parent index per support or None
    """
    J, K = len(X), len(X[0])
    Usets = [set(S) for S in U]

    for s_idx, S in enumerate(Usets):
        p_idx = parent[s_idx]
        if p_idx is None:
            # S is maximal → no C2 clause needed here
            continue

        # only items that "drop out" when going from P to S
        Pa = Usets[p_idx]
        S_jp = (j for j in Pa if j not in S)

        for jp in S_jp:
            witness = []
            for k in range(K):
                c = pool.id(('c', s_idx, jp, k))
                witness.append(c)

                # forward: c → x[jp,k]=1  and  ∀j∈S: x[j,k]=0
                cnf.append([-c, X[jp][k]])
                for j in S:
                    cnf.append([-c, -X[j][k]])

                # backward: (x[jp,k] ∧ ∧_{j∈S} ¬x[j,k]) → c
                cnf.append([-X[jp][k]] + [X[j][k] for j in S] + [c])

            # at least one witnessing column
            cnf.append(witness)
    
    
def solve_SAT(Q, solver_name='cadical195'):
    J, K = Q.shape
    Cardbound = K - distances(Q)
    U = unique_pattern_supports(Q)
    parent = minimal_size_parent(U)
    pool = IDPool()
    X = [[pool.id(('x', j, k)) for k in range(K)] for j in range(J)]
    cnf = CNF()

    add_lex_decreasing(cnf, pool, X, J, K)
    add_neq_Q(cnf, X, Q)
    add_row_cardinality(cnf, pool, X, Cardbound)
    add_U_constraint(cnf, pool, X, U, parent)

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as s:
        is_sat = s.solve()
        print(s.accum_stats())
        if not is_sat:
            return None
        model = set(s.get_model())
        Q_bar = np.zeros_like(Q)
        for j in range(J):
            for k in range(K):
                if X[j][k] in model:
                    Q_bar[j, k] = 1
        return Q_bar
    
    
    
def solve_SAT_fast(Q, solver_name='cadical195'):
    J, K = Q.shape
    U = unique_pattern_supports(Q)
    pool = IDPool()
    X = [[pool.id(('x', j, k)) for k in range(K)] for j in range(J)]
    cnf = CNF()

    add_neq_Q(cnf, X, Q)
    add_leq_Q(cnf, X, Q)
    add_U_constraint(cnf, pool, X, U)

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as s:
        found = s.solve()
        print(s.accum_stats())
        if not found:
            return None
        model = set(s.get_model())
        Q_bar = np.zeros_like(Q)
        for j in range(J):
            for k in range(K):
                if X[j][k] in model:
                    Q_bar[j, k] = 1
        return Q_bar