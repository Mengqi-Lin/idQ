# Data

- `examples/`: tiny binary input matrices used by the demo notebook.
- `raw/`: per-seed experiment CSVs; keep these as the source results.
- `processed/`: summary tables derived from completed raw files.

Use `IDQ_DATA_DIR=/absolute/path/to/data` to direct new experiments and summaries
to another data root, such as cluster scratch storage. This folder does not
include the historical solver benchmark data; those are in the separate
`solver_comparison.zip` archive.
