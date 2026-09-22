# Single-clause exclusion benchmark

163 SAT-stage inputs, 159 distinct basis matrices; 1467 timed runs. Distinct-basis outcomes: {'UNSAT': 148, 'SAT': 11}.

Timings for exactly identical basis matrices are pooled. Each distinct basis is summarized by the median of its repetitions. Total times sum those medians; they are not elapsed experiment time.
The SAT-stage time includes CNF construction, solver initialization, and solve time. Sampling, shared preprocessing, validation, timer management, garbage collection, and solver destruction are excluded.
UNKNOWN runs are counted as timeouts, never as UNSAT. Speedups use only cases where every repeat completed for both compared encodings.

| Encoding | Complete cases | Timeout cases | Median solve (ms) | Total solve (s) | Total SAT stage (s) | Paired aggregate speedup |
|---|---:|---:|---:|---:|---:|---:|
| prefix | 159 | 0 | 7.323 | 2.785 | 3.559 | 1.000 |
| exclude_h | 159 | 0 | 6.971 | 2.510 | 3.264 | 1.090 |
| exclude_x | 159 | 0 | 6.887 | 2.473 | 3.227 | 1.103 |

| Group | Encoding | Cases | Total solve (s) | Total SAT stage (s) |
|---|---|---:|---:|---:|
| SAT | prefix | 11 | 0.330 | 0.368 |
| SAT | exclude_h | 11 | 0.186 | 0.221 |
| SAT | exclude_x | 11 | 0.234 | 0.268 |
| UNSAT | prefix | 148 | 2.455 | 3.192 |
| UNSAT | exclude_h | 148 | 2.324 | 3.044 |
| UNSAT | exclude_x | 148 | 2.239 | 2.959 |
| bernoulli | prefix | 65 | 1.537 | 1.850 |
| bernoulli | exclude_h | 65 | 1.343 | 1.649 |
| bernoulli | exclude_x | 65 | 1.334 | 1.638 |
| fixed_weight | prefix | 66 | 1.114 | 1.488 |
| fixed_weight | exclude_h | 66 | 1.052 | 1.420 |
| fixed_weight | exclude_x | 66 | 1.031 | 1.398 |
| mixed_sparse | prefix | 14 | 0.066 | 0.105 |
| mixed_sparse | exclude_h | 14 | 0.048 | 0.084 |
| mixed_sparse | exclude_x | 14 | 0.049 | 0.086 |
| structured | prefix | 14 | 0.068 | 0.117 |
| structured | exclude_h | 14 | 0.066 | 0.112 |
| structured | exclude_x | 14 | 0.058 | 0.105 |

Environment and source hashes are in metadata.json. Exact inputs are in the corpus cases.json. Every SAT witness is in runs.jsonl and was independently checked using Boolean multiplication and column-permutation comparison.
