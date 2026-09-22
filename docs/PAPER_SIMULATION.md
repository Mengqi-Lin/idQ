# Rerun the revised paper's simulation study

Use idQ 0.1.3 and the dedicated paper scripts below. They reproduce the **design**
in the supplied simulation section and generate fresh, reproducible matrices.
They do not promise the exact old percentages: the old raw matrices and original
seed schedule were not supplied. Save this run's raw files to retain its exact
matrices for future comparisons.

## What runs

| Design | Settings | Replicates |
| --- | --- | --- |
| Bernoulli | J=25,50,100; K=5,10; p=0.1,0.3,0.5,0.7,0.9 | 30 settings × 1,000 |
| Sparsity | J=25,50,100; K=10; row size uniform on 1,...,m; m=3,4 | 6 settings × 1,000 |

All 36,000 matrices use the conjunctive algorithm with basis reduction,
identity/two-column/three-column checks, Glucose42, `exclude_x`, weak column
ordering, and `maximal_candidate=False`. Bernoulli entries are **unrestricted iid**:
zero rows are allowed during generation and removed by basis reduction. The
generic Bernoulli CLI has a different default, so use this paper submission path.

The default splits each setting into 10 tasks of 100 matrices: 360 array tasks,
with at most 20 running at once. Each task is serial and requests one CPU, 4 GB,
and 4 hours on `standard`. BLAS/OpenMP threads are limited to one, and `srun`
binds the worker to the allocated core. These are resource requests, not runtime
predictions. A separate analysis job requests one CPU, 4 GB, and one hour.

Base seed 20260922 is combined with the setting index and batch index using
NumPy SeedSequence. Each task receives a recorded integer seed for its local
MT19937 RandomState. Changing the batch size changes the sample; changing only
array concurrency does not.

## Install on Great Lakes

Place the updated package in your existing `idQ` directory. The archive contains
code and example/benchmark files; extracting it does not delete other files in
your data directories. Run from the directory that contains `pyproject.toml`:

```bash
cd ~/idQ
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -c 'import idQ; from pysat.solvers import Glucose42; print(idQ.__version__, idQ.__file__); Glucose42().delete()'
```

Use an available Python 3.10 or later; load your usual Python module first if
needed. The output should identify version 0.1.3 in this package directory.
If you already have a working environment, keep it and reinstall the updated
package there. Set `IDQ_PYTHON` to its Python executable when it is not `.venv/bin/python`.
PySAT supplies Glucose42; no external solver executable or license is required.

## Submit the full study

Run the helper with **bash from a Great Lakes login-node terminal**. It calls
`sbatch` itself; do not submit this helper with `sbatch`.

```bash
cd ~/idQ
bash jobs/submit_paper_simulation.sh --dry-run data/paper_final
bash jobs/submit_paper_simulation.sh data/paper_final
```

If your usual jobs specify a Slurm account, use that same account:

```bash
export IDQ_ACCOUNT=your_actual_slurm_account
bash jobs/submit_paper_simulation.sh data/paper_final
```

Choose one of the submission commands, not both. The dry run creates the study
manifest and shows the two `sbatch` commands without submitting either job.
The actual submission prints an array ID and an analysis-job ID. Analysis starts
only after all array tasks succeed. If that dependency becomes impossible, Slurm
cancels the dependent analysis job; a retry submission queues a new one.

`data/paper_final` is a new, isolated result directory. An absolute path on your
account's scratch storage also works. Do not modify package source or change
software versions while this study is running: workers compare the source with
the prepared manifest, and analysis rejects mixed source/software versions.

To change resource requests or concurrency before submission:

```bash
IDQ_MAX_CONCURRENT=10 IDQ_TIME=08:00:00 IDQ_MEM=8G \
  bash jobs/submit_paper_simulation.sh data/paper_final
```

Other options are `IDQ_PARTITION`, `IDQ_CONSTRAINT` (an existing cluster CPU
feature, if you want one processor type), and `IDQ_AUTO_ANALYZE=0` to skip the
automatic analysis job. `IDQ_REPLICATES`, `IDQ_PER_TASK`, and `IDQ_BASE_SEED` are
used only when preparing a **new** study directory. Existing manifests define
their own sample size and seeds.

## Monitor and retry

```bash
squeue -u "$USER"
.venv/bin/python -m idQ.experiments.paper_study status --study-dir data/paper_final
```

The local status command checks output-file presence; the analysis command
performs the full validation. Slurm output/error logs and submitted job IDs are
saved under `data/paper_final/logs/` and `data/paper_final/submissions.tsv`.
For a reported array ID, inspect its resource use and failures with:

```bash
sacct -j ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

After the old array has stopped, resubmit missing or interrupted tasks:

```bash
bash jobs/submit_paper_simulation.sh --missing data/paper_final
```

This keeps completed files and reruns unfinished tasks with the same seeds.
Partial attempts are moved into `attempts/` before retrying. Increase memory or
walltime in that command if `sacct` reports an out-of-memory failure or timeout.
A task killed before all its matrices finish is unresolved; missing observations
are never silently counted as identifiable or nonidentifiable.

## Analyze and update the manuscript

The analysis job normally does this automatically. Its equivalent Python command
is:

```bash
.venv/bin/python -m idQ.experiments.paper_summary --study-dir data/paper_final
```

To rerun analysis on a compute node, from `~/idQ`:

```bash
sbatch --output="$PWD/data/paper_final/logs/analysis_%j.out" \
       --error="$PWD/data/paper_final/logs/analysis_%j.err" \
       jobs/analyze_paper_simulation.sh "$PWD/data/paper_final" "$PWD"
```

Add `--account=your_actual_slurm_account` if required. The analysis checks every
expected task, all 1,000 replicates in each setting, the recorded Q matrices and
indicators, solver/encoding/software/source settings, and duplicate or incomplete
files. It will not produce final paper tables from an incomplete study.

| Output under `analysis/` | Use |
| --- | --- |
| `paper_tables.tex` | Replacement tables with labels `tab:bern`, `tab:bern-pure-node-gap`, `tab:sparse`, and `tab:bern-computation` |
| `cell_summary.csv` | Exact numerators, denominators, percentages, branch counts, and unrounded runtime means for all 36 settings |
| `summary.json` | Completion status, all numerical summaries, and provenance |
| `runtime_environment.json` | Software versions, CPU models, hostnames, allocation/thread settings, and timing definitions for the completed jobs |
| `README.md` | Interpretation of the outputs |

Update the numerical statements in the results paragraphs from these new
summaries as well as replacing the tables. Do not reuse the old runtime prose.
Conditional proportions with no identifiable matrices are undefined and appear
as dashes. Runtime values below 0.001 seconds are displayed as `<0.001`, while
unrounded values remain in the CSV/JSON.

For a clearly marked interim summary while jobs are running:

```bash
.venv/bin/python -m idQ.experiments.paper_summary \
    --study-dir data/paper_final --allow-partial
```

Those outputs go into `analysis_partial/` and are not final paper results.

## What the reported times mean

`preprocess_time` sums the timed basis reduction and the checks actually executed
in Steps 0--1. `algorithm_time` measures wall-clock time inside `identify`, after
input/option validation, through the returned answer. It includes CNF construction,
solver setup and solving, plus certificate validation/lifting when a SAT witness
is returned. It excludes sampling, extra simulation diagnostics, metadata, and
CSV writing. `sat_time` is the SAT path's elapsed time, not solver-only time.

All means in the computational table average over **all 1,000 matrices** in each
setting, with zero SAT time when SAT is skipped. They are not means conditional
on invoking SAT. The K=10 runtime table uses the same Bernoulli observations as
the identifiability tables, so no separate runtime experiment is needed.

Every raw CSV has a `.metadata.json` sidecar recording the actual processor,
hostname, Python/NumPy/PySAT/idQ versions, SLURM allocation, source hashes,
settings, and timing definitions. Raw CSVs store each original Q as a bitstring
and SHA256 fingerprint. Preserve the complete study folder, not just its summaries.
Different processor models are reported explicitly; review that information before
describing the hardware in the paper or attributing differences from old timings
solely to the new encoding.

## Small end-to-end check

For an optional cluster check of all 36 settings with two matrices each:

```bash
IDQ_REPLICATES=2 IDQ_PER_TASK=2 IDQ_MAX_CONCURRENT=4 \
  bash jobs/submit_paper_simulation.sh data/paper_smoke
```

Its 72 matrices produce `analysis_smoke/paper_tables_SMOKE.tex`, visibly marked
as test results. Use a different directory, with default settings, for the full
36,000-matrix study.

## References for cluster commands

- [U-M: Submit a job](https://documentation.its.umich.edu/node/5066)
- [Slurm: Job arrays](https://slurm.schedmd.com/job_array.html)
- [Slurm: CPU binding](https://slurm.schedmd.com/resource_binding.html)
