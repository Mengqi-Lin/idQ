# Verification — idQ 0.1.1, 16 September 2026

This is a historical validation record. For the finalized single-clause default
and paper-study workflow in version 0.1.3, see
[VERIFICATION_PAPER_STUDY.md](VERIFICATION_PAPER_STUDY.md).

- All 42 package regression tests pass (30.3 seconds on the preparation machine).
- New tests compare all fixed H assignments through K=3 for every exposed
  cardinality encoding, check SAT/UNSAT certificates with and without symmetry
  and maximal-candidate constraints, guard auxiliary variable allocation, and
  check the quadratic literal budget of the default.
- An independent audit checked the default cardinality predicate against every
  H through K=4. The existing suite also compares complete factorization CNFs,
  the profile encoding, and a direct oracle on 119 small antichain matrices.
- K=1's impossible factor-cardinality condition was checked on Glucose42,
  CaDiCaL195, CaDiCaL300, Kissat404, and MapleChrono using contradictory units.
- All 11 notebook code cells execute with IPython and real Glucose 4.2;
  saved outputs include the new default and an explicit Sinz-counter example.
  The preparation environment prohibits local sockets, so notebook execution
  uses a single IPython process rather than a socket-based Jupyter kernel.
- The default production CNFs match the benchmark prefix-witness CNFs exactly
  on all 13 saved matrices, including variable numbering and clause order.
- The updated LaTeX subsection passes standalone syntax compilation with
  stubs for the surrounding manuscript macros, references, and citations.
- Local experiment workers and the CSV summary workflow passed integration
  checks. Previous shell syntax and mocked Slurm checks remain applicable;
  no job scripts changed in this release and no cluster jobs were submitted.

The separate encoding comparison archive contains raw measurements, exact
matrices, the source snapshot, reproducible commands, and detailed limitations.
Its benchmark harness runs solvers serially. Timings were collected in a shared
execution environment and do not predict Great Lakes runtimes. Timeouts remain
unresolved; SAT certificates were independently checked, while UNSAT outputs
are solver conclusions without independently checked proof certificates.
