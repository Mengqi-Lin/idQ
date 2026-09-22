# Single-clause exclusion benchmark

16 SAT-stage inputs, 16 distinct basis matrices; 144 timed runs. Distinct-basis outcomes: {'SAT': 16}.

Timings for exactly identical basis matrices are pooled. Each distinct basis is summarized by the median of its repetitions. Total times sum those medians; they are not elapsed experiment time.
The SAT-stage time includes CNF construction, solver initialization, and solve time. Sampling, shared preprocessing, validation, timer management, garbage collection, and solver destruction are excluded.
UNKNOWN runs are counted as timeouts, never as UNSAT. Speedups use only cases where every repeat completed for both compared encodings.

| Encoding | Complete cases | Timeout cases | Median solve (ms) | Total solve (s) | Total SAT stage (s) | Paired aggregate speedup |
|---|---:|---:|---:|---:|---:|---:|
| prefix | 16 | 0 | 6.832 | 0.113 | 0.179 | 1.000 |
| exclude_h | 16 | 0 | 4.957 | 0.087 | 0.150 | 1.193 |
| exclude_x | 16 | 0 | 4.540 | 0.086 | 0.149 | 1.202 |

| Group | Encoding | Cases | Total solve (s) | Total SAT stage (s) |
|---|---|---:|---:|---:|
| SAT | prefix | 16 | 0.113 | 0.179 |
| SAT | exclude_h | 16 | 0.087 | 0.150 |
| SAT | exclude_x | 16 | 0.086 | 0.149 |
| planted_sat | prefix | 16 | 0.113 | 0.179 |
| planted_sat | exclude_h | 16 | 0.087 | 0.150 |
| planted_sat | exclude_x | 16 | 0.086 | 0.149 |

Environment and source hashes are in metadata.json. Exact inputs are in the corpus cases.json. Every SAT witness is in runs.jsonl and was independently checked using Boolean multiplication and column-permutation comparison.
