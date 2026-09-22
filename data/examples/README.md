# Example matrices

These CSV files have no header or row labels. Each line is a row of a binary Q matrix.

- `Q_identifiable_no_pure.csv`: a 5-by-4 matrix satisfying the algebraic condition with no pure nodes; it reaches SAT.
- `Q_nonidentifiable.csv`: a 5-by-4 matrix failing the algebraic condition; SAT returns a factorization certificate.

Both matrices appear in `notebooks/idQ_demo.ipynb`. They are deterministic examples, not simulation outputs.
