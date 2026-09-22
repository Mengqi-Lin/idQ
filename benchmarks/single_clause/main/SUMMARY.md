# Single-clause exclusion benchmark

163 SAT-stage inputs, 159 distinct basis matrices; 1467 timed runs. Distinct-basis outcomes: {'UNSAT': 148, 'SAT': 11}.

Timings for exactly identical basis matrices are pooled. Each distinct basis is summarized by the median of its repetitions. Total times sum those medians; they are not elapsed experiment time.
The SAT-stage time includes CNF construction, solver initialization, and solve time. Sampling, shared preprocessing, validation, timer management, garbage collection, and solver destruction are excluded.
UNKNOWN runs are counted as timeouts, never as UNSAT. Speedups use only cases where every repeat completed for both compared encodings.

| Encoding | Complete cases | Timeout cases | Median solve (ms) | Total solve (s) | Total SAT stage (s) | Paired aggregate speedup |
|---|---:|---:|---:|---:|---:|---:|
| prefix | 159 | 0 | 6.953 | 20.387 | 21.093 | 1.000 |
| exclude_h | 159 | 0 | 5.042 | 17.389 | 18.078 | 1.167 |
| exclude_x | 159 | 0 | 5.622 | 17.249 | 17.943 | 1.176 |

| Group | Encoding | Cases | Total solve (s) | Total SAT stage (s) |
|---|---|---:|---:|---:|
| SAT | prefix | 11 | 2.554 | 2.588 |
| SAT | exclude_h | 11 | 0.581 | 0.614 |
| SAT | exclude_x | 11 | 0.621 | 0.655 |
| UNSAT | prefix | 148 | 17.833 | 18.505 |
| UNSAT | exclude_h | 148 | 16.808 | 17.464 |
| UNSAT | exclude_x | 148 | 16.628 | 17.288 |
| bernoulli | prefix | 65 | 14.846 | 15.132 |
| bernoulli | exclude_h | 65 | 12.534 | 12.814 |
| bernoulli | exclude_x | 65 | 12.256 | 12.542 |
| fixed_weight | prefix | 66 | 5.413 | 5.753 |
| fixed_weight | exclude_h | 66 | 4.791 | 5.125 |
| fixed_weight | exclude_x | 66 | 4.923 | 5.256 |
| mixed_sparse | prefix | 14 | 0.061 | 0.094 |
| mixed_sparse | exclude_h | 14 | 0.026 | 0.058 |
| mixed_sparse | exclude_x | 14 | 0.030 | 0.062 |
| structured | prefix | 14 | 0.067 | 0.113 |
| structured | exclude_h | 14 | 0.038 | 0.081 |
| structured | exclude_x | 14 | 0.040 | 0.083 |

Environment and source hashes are in metadata.json. Exact inputs are in the corpus cases.json. Every SAT witness is in runs.jsonl and was independently checked using Boolean multiplication and column-permutation comparison.
