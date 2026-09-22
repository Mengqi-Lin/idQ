# SAT encoding in idQ

The finalized default uses **Glucose 4.2** with `cardinality_encoding="exclude_x"`.
It encodes the Boolean factorization `Q = Q_bar ⊙ H`, places the columns of
`Q_bar` in weakly nonincreasing lexicographic order, and adds one clause
excluding the sorted copy of `Q`. A satisfying assignment supplies a matrix
that is not equivalent to `Q` up to column permutation. The returned
factorization and non-equivalence are independently verified.

## Single-clause non-equivalence

Let `Q_sorted` be `Q` with its columns sorted in the same lexicographic order
as `Q_bar`. The default clause contains `x[j,l]` where `Q_sorted[j,l]` is zero
and `-x[j,l]` where it is one. This is exactly `Q_bar != Q_sorted`: one clause
of length `J*K`, with no auxiliary variables. Since `Q_bar` is sorted, this
excludes all matrices equivalent to `Q`.

The implementation factorizes `Q` in its supplied column order and uses its
sorted copy only in the exclusion clause. This is equivalent to sorting `Q`
before the paper's SAT construction, while preserving the original column
coordinates of the returned factorization.

Both single-clause options require `symmetry_breaking=True`. Weak ordering
permits equal candidate columns. The existing two-column prerequisite, the
factorization clauses, and the implied nonzero candidate-row and nonempty
`H`-column constraints remain in place. `maximal_candidate=False` remains the
default.

The alternative `exclude_h` requires an entry of `H` outside the unique
permutation `P` satisfying `Q = Q_sorted ⊙ P`. Under the two-column condition,
the sorted candidate is equivalent to `Q` if and only if `H=P`. For `K>=2`,
this alternative exclusion clause has `K*(K-1)` positive literals.

## Encoding size

For `K>=2`, the following counts include the `K` implied nonempty `H`-column
clauses. They exclude factorization and lexicographic constraints.

| Encoding | Auxiliary variables | Clauses | Literal occurrences |
| --- | ---: | ---: | ---: |
| Default `exclude_x` | 0 | K+1 | JK+K² |
| Optional `exclude_h` | 0 | K+1 | 2K²−K |
| Earlier `prefix` | K(2K−3) | 5K²−7K+1 | 13K²−19K |
| Earlier `prefix_oneway` | K(2K−3) | 3K²−3K+1 | 9K²−11K |
| Earlier `legacy` | K | K²+K+1 | K³+K²+K |

For a positive entry of `Q`, product witnesses use two implication clauses
per term plus one witness disjunction: `2K+1` clauses. The reverse AND
implication is unnecessary. If `s` is the number of ones in `Q`, the
factorization uses `Ks` auxiliary variables, `JK²+(K+1)s` clauses, and
`2JK²+3Ks` literal occurrences. Including the matrix variables and the
`O(JK)` lexicographic encoding, the full formulation has size `O(JK²)`.
These are encoding-size bounds, not solve-time predictions.

## Reproducing earlier encodings

```python
from idQ import identify

result = identify(Q)  # exclude_x, glucose42, maximal_candidate=False
prefix_result = identify(Q, cardinality_encoding="prefix")
legacy_result = identify(Q, cardinality_encoding="legacy")
print(result.sat_cardinality_encoding)
```

Available choices are `exclude_x`, `exclude_h`, `prefix`, `prefix_oneway`,
`seqcounter`, `cardnetwrk`, `totalizer`, and `legacy`. The older encodings
remain explicit options for reproducible comparisons. To disable symmetry
breaking in a direct `solve_sat` call, explicitly select an older cardinality
encoding such as `prefix`.

`prefix` defines exact prefix ORs and uses implication-only second-one
witnesses. `prefix_oneway` also relaxes the prefix gates to one-way witness
implications. `legacy` preserves the older gadget and its clause order.
The general encodings use PySAT `CardEnc.atleast` on the `K²` entries of `H`
with bound `K+1`; `seqcounter` selects Sinz's sequential counter. Auxiliary
variable identifiers are reserved to avoid collisions with later constraints.

The result field `sat_cardinality_encoding` is `None` if preprocessing finishes
without SAT. Keep separate run directories when comparing encodings or
package versions. Historical benchmarks of the earlier cardinality encodings
describe those configurations and do not establish a universal runtime ranking
for the finalized default.

## References

- [PySAT cardinality API](https://pysathq.github.io/docs/html/api/card.html)
- [Sinz (2005), Towards an Optimal CNF Encoding of Boolean Cardinality Constraints](https://www.carstensinz.de/papers/CP-2005.pdf)
