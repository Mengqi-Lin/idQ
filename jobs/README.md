# Cluster jobs

For the finalized paper's full Bernoulli and sparsity grids, use
`bash jobs/submit_paper_simulation.sh STUDY_DIR` and follow
[the paper rerun guide](../docs/PAPER_SIMULATION.md). That worker requests
one CPU, 4 GB and 4 hours on `standard`; its defaults are separate from the
older per-design workers documented below.

These scripts run the Bernoulli and row-sparsity experiments on Great Lakes.
Glucose 4.2 (`glucose42`) is the default solver. Each worker requests one CPU,
56 GB total memory, and 68 hours on `largemem,standard`, preserving the total
memory and time from the old scripts. Edit the `#SBATCH` lines in the worker
scripts if your cluster allocation needs different resources. No Gurobi module
or license is needed.

## Set up Python once

From the project directory, create an environment using an available Python
3.10 or later. If Great Lakes requires a Python module, load the module used by
your account before these commands.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Workers use `.venv/bin/python` when it exists. To use a different installed
environment, set `IDQ_PYTHON` to its Python executable before submitting. This
is one executable path, not a shell command or a list of Python arguments.
Slurm inherits the login-shell environment; any required modules should be
loaded before submission.

## Preview and submit

Run the **submit helpers with `bash` on the login node**, rather than submitting
the helpers themselves with `sbatch`. Each helper creates the output directories
and submits one array with seeds `0` through `nseeds - 1`. Each array task runs
`N=10` matrices by default.

```bash
# Bernoulli: J K p nseeds [solver]
bash jobs/submit_idQ_expr.sh --dry-run 50 10 0.3 100
IDQ_MAX_CONCURRENT=20 bash jobs/submit_idQ_expr.sh 50 10 0.3 100

# Row sparsity: J K m nseeds [solver [min_row_size [distribution [probs]]]]
bash jobs/submit_row_sparsity_expr.sh --dry-run 50 10 3 100
bash jobs/submit_row_sparsity_expr.sh 50 10 3 100

# Exclude pure nodes: row sizes are uniform on {2,3,4}.
bash jobs/submit_row_sparsity_expr.sh 50 10 4 100 glucose42 2

# Custom probabilities on row sizes {1,2,3}.
bash jobs/submit_row_sparsity_expr.sh 50 10 3 100 glucose42 1 custom 0.4,0.4,0.2

# Change matrices per seed, array concurrency, and output storage.
IDQ_N=25 IDQ_MAX_CONCURRENT=10 IDQ_DATA_DIR=/scratch/your-account/idQ-data \
    bash jobs/submit_idQ_expr.sh 100 20 0.3 40
```

`--dry-run` prints a shell-quoted `sbatch` command and creates output directories,
but never submits a job. The optional solver argument is forwarded to the
experiment CLI. Prefer named solvers such as `glucose42` or `cadical195`.

The worker commands keep the old positional order, with the solver now
optional. A tiny local smoke run is:

```bash
bash jobs/idQ_expr.sh 6 3 1 0.3 0
bash jobs/row_sparsity_expr.sh 6 3 1 2 0
```

These run Python directly on the current machine. Submit larger experiments
through the array helpers. The helpers supply the special seed `array`; the
worker replaces it with `SLURM_ARRAY_TASK_ID`. The package root is exported as
`IDQ_ROOT` and passed as Slurm's working directory, so workers still locate the
project when Slurm executes a copied script from its spool directory.

## Outputs and environment variables

Each seed writes its own CSV, so array tasks do not append to the same file:

```text
data/raw/bernoulli/run_<array-job-id>/bernoulli_seed_0_diag.csv
data/raw/row_sparsity/run_<array-job-id>/rowsparse_seed_0_diag.csv
logs/bernoulli_<array-job-id>_<task-id>.out
logs/bernoulli_<array-job-id>_<task-id>.err
```

Row-sparsity logs use `row_sparsity` in place of `bernoulli`. Local worker runs
use a timestamp and process ID in the run directory. Requeuing a Slurm task
targets the same CSV for that seed. An existing completed CSV is not overwritten
automatically; submitting a new array creates a new run directory.

After a row-sparsity array finishes, summarize its CSVs with:

```bash
.venv/bin/python -m idQ.experiments.summary \
    --data-dir data/raw/row_sparsity/run_123456 \
    --output-csv data/summaries/row_sparsity_123456.csv
```

Replace `123456` with the numeric ID returned by `sbatch`. Use the actual
data path when `IDQ_DATA_DIR` points elsewhere. The summary command validates
complete result files and checks for duplicate replicates. Its default search
is recursive; select a single run directory when you intend to summarize one
submission.

| Variable | Default | Purpose |
| --- | --- | --- |
| `IDQ_N` | `10` | Number of matrices per seed for submit helpers |
| `IDQ_MAX_CONCURRENT` | Unset | Maximum simultaneously running array tasks |
| `IDQ_DATA_DIR` | `<project>/data` | Data root; CSVs go under `raw/`; relative paths resolve from the project |
| `IDQ_PYTHON` | `<project>/.venv/bin/python`, then `python3` | Python executable with idQ installed |
| `IDQ_ROOT` | Detected project directory | Project location exported by submit helpers |

The scripts set common BLAS/OpenMP thread counts to one to match the one-CPU
request. Check your account's array-size and resource limits when choosing
`nseeds`, memory, and concurrency.
