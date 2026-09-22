# K=10 single-clause exclusion benchmark

The single-clause exclusions are worthwhile: both simplify the existing encoding and reduced aggregate SAT-stage time by about 14–15% in the main suite. Their performances were close; this benchmark does not establish a clear winner between excluding the sorted Q matrix (`exclude_x`) and excluding the unique feasible permutation of H (`exclude_h`).

The existing `maximal_candidate=True` option produces a much larger reduction in the slower cases. It increases costs on many easy cases, however, including the separate constructed SAT suite. For deciding existence, both optional changes are mathematically justified. The package default remains `prefix`, with maximal-candidate constraints off; the new exclusions are selectable options in this benchmark snapshot.

## Controlled comparison

- Source: the supplied idQ 0.1.1 archive. Only optional exclusion choices and their tests/documentation were added to package source. The original prefix implementation and default solver remain unchanged.
- Solver: Glucose42 through python-sat 1.9.dev15; Python 3.12.14; NumPy 2.3.5; Linux x86_64.
- All inputs have K=10. All solvers run serially, pinned to one available CPU, with fresh solver instances, three repetitions, and rotated/randomized encoding order. Maximal-candidate off/on were separate serial passes; comparisons between them were not interleaved.
- Basis reduction and the identity, two-column, and three-column checks run before SAT. Every variant receives exactly the same resulting basis, ordering, variable numbering for shared variables, factorization clauses, lexicographic clauses, and nonempty-H-column clauses.
- Five-second per-solve limits. Interrupted/unknown results would not count as UNSAT. No timed run reached its limit.
- Every SAT assignment was independently checked by Boolean multiplication, the H cardinality threshold, and comparison of actual column multisets. All encodings and maximal-candidate settings agreed. Agreement is not a separately checked UNSAT proof.
- Eleven focused correctness tests passed before benchmarking: six new single-clause tests and five existing cardinality tests. These include small independent profile/oracle comparisons, permuted input columns, symmetry guards, certificate validation, and exact encoding-size differences.

## Main corpus

There are 168 seeded random inputs and 18 structured inputs, all retained in `cases.json`:

- Bernoulli entries, conditional on each row being nonzero: J in {20,35,50,80}, p in {0.3,0.5,0.7}, six draws per cell (72 inputs).
- Fixed row weights in {2,3,5}, the same four J values, six draws per cell (72 inputs).
- Uniform row weights 1 through 3, the same four J values, six draws per cell (24 inputs).
- Six structured families with three row/column permutations each: cycle C10; all weight-two rows; all weight-three rows; the six weight-two rows on four coordinates combined with an identity on six coordinates; C10 plus its diameter matching; Petersen graph (18 inputs).

Of these 186 inputs, six exit at completeness, 15 at the two-column check, and two at the three-column check. The remaining 163 inputs reach SAT. They yield 159 distinct basis matrices: all-pairs and all-triples permutations each duplicate the same basis three times. Their timing repetitions are pooled, and each distinct basis receives one unit of weight in the results below. Basis row counts range from 10 to 120. Outcomes are 148 UNSAT and 11 SAT distinct bases.

Main-suite SAT-stage timings:

| Encoding | Median, maximal off (ms) | Total, maximal off (s) | Median, maximal on (ms) | Total, maximal on (s) |
|---|---:|---:|---:|---:|
| `prefix` | 11.12 | 21.09 | 11.62 | 3.56 |
| `exclude_x` | 9.85 | 17.94 | 11.65 | 3.23 |
| `exclude_h` | 9.70 | 18.08 | 11.61 | 3.26 |

A total is the sum of per-basis median times, not the elapsed duration of the experiment. SAT-stage time includes CNF construction, solver initialization, and solving. It excludes sampling, shared preprocessing, certificate validation, garbage collection, timer management, and solver destruction. These are not complete public `identify` timings.

With maximal-candidate off, `exclude_x` reduces total time from 21.09 s to 17.94 s (14.9% reduction), and `exclude_h` to 18.08 s (14.3% reduction). The former is faster on 125/159 bases, the latter on 127/159. Neither dominates the prefix method on every instance. Median SAT-stage time falls from 11.12 ms to 9.85 ms or 9.70 ms.

The large maximal-candidate benefit is concentrated in slower cases. Combining `exclude_x` with maximal-candidate reduces total time to 3.23 s, about 6.54 times faster than the current default on this suite. But its median is 11.65 ms, slightly above the current default's 11.12 ms. Relative to `exclude_x` alone, maximal-candidate speeds up only 69/159 bases while strongly reducing the slow tail. For example, one J=50,p=0.7 UNSAT input falls from 3.119 s with the current default to 0.048 s with `exclude_x` plus maximal-candidate; a J=35,p=0.7 SAT input falls from 2.481 s to 0.089 s.

## Separate constructed SAT corpus

This supplies additional non-identifiable instances independently of solver outcomes. Let H have the ten edge-incidence rows of C10. For each J in {20,35,50}, generate six matrices X whose rows select exactly two factors uniformly, and set Q=X odot H. The known witnesses are saved. All 18 draws are retained; two fail the two-column check and 16 reach SAT. All 16 are SAT under every tested setting.

| Encoding | Total SAT stage, maximal off (s) | Total SAT stage, maximal on (s) |
|---|---:|---:|
| `prefix` | 0.1523 | 0.1786 |
| `exclude_x` | 0.1315 | 0.1486 |
| `exclude_h` | 0.1304 | 0.1498 |

Here each input is easy. Single-clause exclusion improves the aggregate by about 14%, while maximal-candidate adds cost relative to the same exclusion without closure. This is why the maximal-candidate result should be described as a substantial improvement on the hard cases in the tested main corpus, not a uniform runtime improvement.

## Encoding change

`exclude_x` retains full weak non-increasing lexicographic ordering of the candidate matrix and excludes the sorted copy of Q using one clause. `exclude_h` uses one positive disjunction of H entries outside the unique permutation mapping the sorted Q back to Q. Both preserve the current implied nonempty-column clauses for H. Both reject `symmetry_breaking=False`.

At K=10 either variant removes 170 auxiliary variables and 420 clauses relative to the prefix encoding. The X clause has 10 J_b literals; the H clause has 90 literals. The general O(J_b K^2) encoding-size order does not change. Existing maximal-candidate constraints add J_b K clauses and no variables.

The attached package is still named `idQ`; solver defaults and existing experiment commands are unchanged. New options:

```python
from idQ import identify

result = identify(Q, cardinality_encoding="exclude_x")
result = identify(Q, cardinality_encoding="exclude_h")
result = identify(Q, cardinality_encoding="exclude_x", maximal_candidate=True)
```

## Recommendation

Use the one-clause exclusion as a serious replacement candidate for the sequential cardinality construction. `exclude_x` offers particularly direct exposition and performs comparably to `exclude_h`. For decision-only workloads where the harder instances drive total computation, maximal-candidate is also useful. Retain the ability to disable it: the tested easy SAT instances do not benefit.

These results cover K=10 and the explicitly described inputs. They support a modest measured improvement for single-clause exclusion, not a general complexity improvement or a claim that one variant is always fastest. Timings are specific to this execution environment.

## Reproduction and files

From the extracted `idQ` directory, using an environment with its requirements installed:

```bash
python -m pip install -e .
python benchmarks/single_clause/run_benchmark.py --cases benchmarks/single_clause/cases.json --out benchmarks/single_clause/replay_main --repeats 3 --timeout 5
python benchmarks/single_clause/analyze.py benchmarks/single_clause/replay_main
python benchmarks/single_clause/run_benchmark.py --cases benchmarks/single_clause/cases.json --out benchmarks/single_clause/replay_maximal --repeats 3 --timeout 5 --maximal
python benchmarks/single_clause/analyze.py benchmarks/single_clause/replay_maximal
python benchmarks/single_clause/run_benchmark.py --cases benchmarks/single_clause/planted_cases.json --out benchmarks/single_clause/replay_planted --repeats 3 --timeout 5
python benchmarks/single_clause/analyze.py benchmarks/single_clause/replay_planted
python benchmarks/single_clause/run_benchmark.py --cases benchmarks/single_clause/planted_cases.json --out benchmarks/single_clause/replay_planted_maximal --repeats 3 --timeout 5 --maximal
python benchmarks/single_clause/analyze.py benchmarks/single_clause/replay_planted_maximal
```

Use a fresh output directory each time. Saved corpus files make the exact matrices reproducible without resampling. The runner generates the primary corpus when its requested cases file does not exist; `make_planted_cases.py` regenerates the separate constructed suite from its fixed seed.

Each run directory contains raw `runs.jsonl` (including all SAT witnesses), environment/source metadata, a machine-readable summary, per-basis medians, and a Markdown summary. `provenance.json` records source-archive, harness, analysis, and corpus hashes. `single_clause.patch` contains the package-source changes. The `pilot` directory is exploratory and is excluded from all reported results. Across the four reported studies there are 3,222 timed solves and zero disagreements or timeouts.
