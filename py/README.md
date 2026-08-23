# idQ simulation files

This folder contains a small refactor of the simulation code.

## Files

- `idq_experiment_helpers.py`  
  Shared identifiability, diagnostics, Q-matrix samplers, and generic simulation loop.

- `idQ_expr.py`  
  Bernoulli-iid simulation driver. This replaces the old monolithic `idQ_expr.py` while keeping the same `run_expr(...)` function name and the same positional CLI interface.

- `row_sparsity_expr.py`  
  New row-sparsity simulation driver for applied diagnostic settings.

- `summarize_row_sparsity.py`  
  Convenience script that summarizes raw row-sparsity CSV files into the reviewer-facing table quantities.

## Bernoulli simulation

```bash
python idQ_expr.py 50 10 100 0.7 1 1
```

Default output:

```text
../data/raw/solver1_J50_K10_p0.7_seed1_diag.csv
```

## Row-sparsity simulation

Default design:

\[
D_j = \|q_j\|_0 \sim \mathrm{Uniform}\{1,\ldots,m\},
\]

then, conditional on `D_j`, sample `D_j` attributes uniformly without replacement.

For `m = 3`:

```bash
python row_sparsity_expr.py 50 10 100 3 1 -1
```

For `m = 4`:

```bash
python row_sparsity_expr.py 50 10 100 4 1 -1
```

Default output examples:

```text
../data/raw/rowsparse_uniform_solver-1_J50_K10_m3_dmin1_seed1_diag.csv
../data/raw/rowsparse_uniform_solver-1_J50_K10_m4_dmin1_seed1_diag.csv
```

## Excluding pure nodes by construction

If you want every row to involve at least two attributes, run with `--min-row-size 2`:

```bash
python row_sparsity_expr.py 50 10 100 4 1 -1 --min-row-size 2
```

This gives:

\[
D_j \sim \mathrm{Uniform}\{2,\ldots,m\}.
\]

## Fixed row size

If every row should have exactly `m` ones:

```bash
python row_sparsity_expr.py 50 10 100 3 1 -1 --row-size-distribution fixed
```

## Custom row-size distribution

For example, with `m = 3` and support `{1,2,3}`, use probabilities `(0.4,0.4,0.2)`:

```bash
python row_sparsity_expr.py 50 10 100 3 1 -1 \
  --row-size-distribution custom \
  --row-size-probs 0.4,0.4,0.2
```

With `--min-row-size 2` and `m = 4`, the support is `{2,3,4}`. So three probabilities are expected.

## Parallel jobs

The drivers write one file per seed by default and overwrite that seed-specific file unless `--append` is passed. This is intentional: it avoids unsafe parallel appends to one shared CSV.

Do not launch two jobs with the same `(solver, J, K, m/p, min_row_size, seed)` unless you intend the later job to overwrite the earlier seed file.

## Summarize row-sparsity results

After running many seeds, combine and summarize them with:

```bash
python summarize_row_sparsity.py \
  --data-dir ../data/raw \
  --output-csv ../data/processed/row_sparsity_summary.csv \
  --display-csv ../data/processed/row_sparsity_summary_display.csv
```

The summary includes:

- identifiable proportion;
- identity-submatrix/completeness proportion;
- identifiable but not complete proportion;
- not-complete among identifiable matrices;
- identifiable with no pure nodes proportion;
- no-pure-node among identifiable matrices;
- average `J_basis`;
- average `M_basis`;
- SAT-called proportion;
- branch proportions.

## Import requirements

These files assume your existing project modules are importable from the run directory:

- `Qbasis.py`
- `solve_SAT.py`
- `idQ.py`

Put these new files in the same directory as your current `idQ_expr.py`, or run them from a directory where those modules are already on `PYTHONPATH`.
