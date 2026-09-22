# Moving from the flat scripts

The distribution and import name are both **idQ**. Install this directory once
with `python -m pip install -e .`; do not copy individual source files to the
working directory or add `src/idQ` directly to `sys.path`.

| Former import or command | Current location |
| --- | --- |
| `from idQ import identify, identifiability` | Unchanged |
| `from Qbasis import reduce_to_basis` | `from idQ import reduce_to_basis` |
| `from solve_SAT import ...` | `from idQ.sat import ...` |
| `from utils import ...` | `from idQ.utils import ...` |
| `python idQ_expr.py ...` | `python -m idQ.experiments.bernoulli ...` |
| `python row_sparsity_expr.py ...` | `python -m idQ.experiments.row_sparsity ...` |
| `python summarize_row_sparsity.py ...` | `python -m idQ.experiments.summary ...` |

The four familiar cluster script names are retained under `jobs/`. Run submit
helpers with `bash`; they submit one Slurm array. Their default solver is
`glucose42`; omit the old solver argument to accept it. Explicit legacy codes
still have their earlier meanings: `-1` and `0` select exact CaDiCaL195, `1`
selects Glucose4.2, and `2` selects MiniSat22. The former restricted solver path
has not been restored.

The scientific decision procedure and simulation CSV schema remain unchanged
by the file reorganization. Output folders and summary discovery were updated
together so the summary command can read new cluster run subfolders.

## Finalized SAT default

The mathematical test and Glucose 4.2 default are unchanged. The default is
now `cardinality_encoding="exclude_x"`: a single clause excludes the sorted
copy of `Q` from the lexicographically ordered candidate. This option requires
weak lexicographic symmetry breaking. `maximal_candidate=False` remains the
default. The code preserves the supplied column coordinates in its returned
factorization, which is equivalent to sorting `Q` before the paper's encoding.

To reproduce an earlier construction, explicitly select `prefix`,
`prefix_oneway`, or `legacy`; `seqcounter` selects PySAT's Sinz counter.
Direct calls that disable symmetry breaking must select an older cardinality
encoding explicitly. See `SAT_ENCODING.md` for the formulations and sizes.

SAT results and experiment CSVs now record `sat_cardinality_encoding`. This
is an additional diagnostic column; existing positional experiment and job
commands still work. Install the updated package with
`python -m pip install -e .` and restart an existing notebook kernel to load
the revised code. Keep separate run directories when comparing versions.
