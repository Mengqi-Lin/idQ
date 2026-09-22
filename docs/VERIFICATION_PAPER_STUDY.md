# Verification — idQ 0.1.3, 22 September 2026

This release makes the finalized `exclude_x` formulation the default and adds
the dedicated workflow in [PAPER_SIMULATION.md](PAPER_SIMULATION.md).

## Package tests

All **74 tests passed** in 35.225 seconds in the preparation environment:

```bash
PYTHONPATH="$PWD/src" python -m unittest discover -s tests -v
```

The suite covers SAT semantics, the single-clause exclusion, basis reduction,
experiment settings and recording, reproducible study preparation, and summary
validation. Analysis tests include incomplete studies, inconsistent settings,
duplicate data, conditional denominators, and runtime aggregation.

## Complete small study

The actual worker scripts ran **all 36 settings with two matrices per setting**
(72 matrices total), followed by the actual analysis script. Both SAT and UNSAT
outcomes and all three preprocessing exits occurred. Analysis reported
`smoke_complete`, `paper_final=false`, and 72 of 72 expected observations. It
produced all four requested table templates and the CSV/JSON summaries under
`analysis_smoke/`. These are test results, not paper simulation estimates.

The source tree recorded by that study has SHA256:

```text
fc0f1ada626eb5c0f19b420791ec0d937f167c5fe4aedcd841f021b4e8c246a1
```

This digest covers the Python source files using the provenance procedure in
`src/idQ/experiments/common.py`; it is not the archive checksum.

## Submission scripts

All three paper job scripts passed `bash -n`. A separate audit used mocked
`sbatch`, without submitting cluster jobs, to check:

- The default manifest contains 36,000 matrices in 360 tasks and submits
  `--array 0-359%20`.
- Workers request one CPU, 4 GB, and four hours; analysis requests one CPU,
  4 GB, and one hour.
- The analysis dependency uses the returned array job ID, including when
  `sbatch --parsable` appends a cluster name.
- `--missing` excludes completed tasks and enables retry for unfinished tasks.
- Workers and analysis resolve the correct package and Python source when their
  scripts are copied into an unrelated Slurm spool directory.
- `--dry-run` makes no submission calls.

No full 36,000-matrix study or actual Great Lakes job was run during this
verification. The resource requests should be reviewed against the completed
cluster jobs; the local trial does not establish production runtimes.
