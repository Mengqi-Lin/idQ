#!/usr/bin/env python3
"""Paired, serial Glucose42 benchmark of non-equivalence encodings.

Run from any directory; the package's local src tree is used automatically.
Saved cases include all random draws, even those resolved by preprocessing.
Wall-clock solver limits are UNKNOWN, never UNSAT. Each SAT model is checked.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pysat
from pysat.solvers import Glucose42
from idQ.basis import reduce_to_basis
from idQ.core import (contains_identity_submatrix, first_two_column_violation,
                      first_three_column_violation)
from idQ.experiments.common import sample_bernoulli_q, sample_row_sparse_q
from idQ.sat import build_factorization_sat_instance
from idQ.utils import is_valid_factorization_counterexample


def preprocess(Q):
    t0 = time.perf_counter()
    B = reduce_to_basis(Q).basis
    if len(B) == 0:
        branch = 'empty_basis'
    elif contains_identity_submatrix(B):
        branch = 'complete'
    elif first_two_column_violation(B) is not None:
        branch = 'two_column'
    elif first_three_column_violation(B) is not None:
        branch = 'three_column'
    else:
        branch = 'sat'
    return B, branch, time.perf_counter() - t0


def make_cases(path, replicates, seed):
    rng = np.random.default_rng(seed)
    cases = []
    def add(Q, family, cell, rep):
        B, branch, elapsed = preprocess(Q)
        digest = hashlib.sha256(np.asarray(Q, dtype=np.uint8).tobytes()).hexdigest()
        cases.append(dict(case_id=f'c{len(cases):04d}', family=family, cell=cell,
                          replicate=rep, J=len(Q), K=Q.shape[1], Jb=len(B),
                          branch=branch, preprocessing_seconds=elapsed,
                          sha256=digest, Q=Q.tolist(), basis=B.tolist()))
    for J in (20, 35, 50, 80):
        for p in (.3, .5, .7):
            for rep in range(replicates):
                add(sample_bernoulli_q(J, 10, p, rng), 'bernoulli', f'J{J}_p{p}', rep)
        for w in (2, 3, 5):
            for rep in range(replicates):
                add(sample_row_sparse_q(J, 10, w, rng, row_size_distribution='fixed'),
                    'fixed_weight', f'J{J}_w{w}', rep)
        for rep in range(replicates):
            add(sample_row_sparse_q(J, 10, 3, rng), 'mixed_sparse', f'J{J}_w1to3', rep)

    # Predeclared structured cases. Permutations exercise ordering sensitivity.
    cycle = np.zeros((10, 10), dtype=int)
    for j in range(10):
        cycle[j, j] = cycle[j, (j + 1) % 10] = 1
    pairs = np.asarray([[int(k in pair) for k in range(10)]
                        for pair in itertools.combinations(range(10), 2)])
    triple_rows = np.asarray([[int(k in t) for k in range(10)]
                              for t in itertools.combinations(range(10), 3)])
    small_pairs = np.asarray([[int(k in pair) for k in range(4)]
                              for pair in itertools.combinations(range(4), 2)])
    block = np.zeros((12, 10), dtype=int)
    block[:6, :4] = small_pairs
    block[6:, 4:] = np.eye(6, dtype=int)
    cubic_edges = [(j, (j + 1) % 10) for j in range(10)] + [(j, j + 5) for j in range(5)]
    cubic = np.asarray([[int(k in edge) for k in range(10)] for edge in cubic_edges])
    petersen_edges = ([(j, (j + 1) % 5) for j in range(5)]
                      + [(j, j + 5) for j in range(5)]
                      + [(j + 5, ((j + 2) % 5) + 5) for j in range(5)])
    petersen = np.asarray([[int(k in edge) for k in range(10)] for edge in petersen_edges])
    for name, Q in [('cycle10', cycle), ('all_pairs', pairs),
                    ('all_triples', triple_rows), ('block_pairs4_identity6', block),
                    ('cubic_cycle_matching', cubic), ('petersen', petersen)]:
        for rep in range(3):
            add(Q[rng.permutation(len(Q))][:, rng.permutation(10)], 'structured', name, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(seed=seed, replicates=replicates, cases=cases), indent=2))
    return cases


def run_one(case, encoding, maximal, timeout):
    B = np.asarray(case['basis'], dtype=int)
    gc.collect()
    begin = time.perf_counter()
    inst = build_factorization_sat_instance(B, cardinality_encoding=encoding,
                                            maximal_candidate=maximal)
    built = time.perf_counter()
    solver = Glucose42(bootstrap_with=inst.clauses)
    loaded = time.perf_counter()
    fired = threading.Event()
    def interrupt():
        fired.set()
        solver.interrupt()
    timer = threading.Timer(timeout, interrupt)
    timer.daemon = True
    timer.start()
    solve_begin = time.perf_counter()
    cpu_begin = time.process_time()
    try:
        answer = solver.solve_limited(expect_interrupt=True)
        cpu_seconds = time.process_time() - cpu_begin
        solved = time.perf_counter()
    finally:
        timer.cancel()
        timer.join()
    stats = solver.accum_stats()
    status = 'UNKNOWN' if answer is None else ('SAT' if answer else 'UNSAT')
    witness = None
    valid = None
    validation_begin = time.perf_counter()
    if answer:
        positive = set(solver.get_model())
        X = np.asarray([[int(v in positive) for v in row] for row in inst.decision_variables])
        H = np.asarray([[int(v in positive) for v in row] for row in inst.factor_variables])
        valid = bool(is_valid_factorization_counterexample(B, X, H))
        if not valid:
            raise RuntimeError(f'Invalid SAT model: {case["case_id"]}, {encoding}')
        witness = dict(X=X.tolist(), H=H.tolist())
    validation_seconds = time.perf_counter() - validation_begin
    solver.delete()
    total = ((built - begin) + (loaded - built) + (solved - solve_begin))
    return dict(case_id=case['case_id'], family=case['family'], cell=case['cell'],
                J=case['J'], K=case['K'], Jb=case['Jb'], encoding=encoding,
                maximal_candidate=maximal, status=status, timeout_seconds=timeout,
                interrupt_fired=fired.is_set(), build_seconds=built-begin,
                load_seconds=loaded-built, solve_seconds=solved-solve_begin,
                solve_cpu_seconds=cpu_seconds, sat_stage_seconds=total,
                validation_seconds=validation_seconds, validated=valid,
                variables=inst.n_variables, clauses=len(inst.clauses),
                literals=sum(map(len, inst.clauses)), solver_stats=stats,
                witness=witness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--replicates', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--encodings', nargs='+', default=['prefix', 'exclude_x', 'exclude_h'])
    parser.add_argument('--maximal', action='store_true')
    parser.add_argument('--limit-cases', type=int)
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    if hasattr(os, 'sched_setaffinity'):
        allowed = os.sched_getaffinity(0)
        chosen = args.cpu if args.cpu in allowed else min(allowed)
        os.sched_setaffinity(0, {chosen})
    args.out.mkdir(parents=True, exist_ok=True)
    cases_path = args.cases or (args.out / 'cases.json')
    if cases_path.exists():
        cases = json.loads(cases_path.read_text())['cases']
    else:
        cases = make_cases(cases_path, args.replicates, args.seed)
    selected = [c for c in cases if c['branch'] == 'sat']
    if args.case_ids:
        selected = [c for c in selected if c['case_id'] in args.case_ids]
    # Shuffle instances independently of solver outcome, before any cap.
    random.Random(args.seed + 123).shuffle(selected)
    if args.limit_cases is not None:
        selected = selected[:args.limit_cases]
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / 'src' / 'idQ').glob('*.py')}
    metadata = dict(python=sys.version, platform=platform.platform(), pysat=pysat.__version__,
                    numpy=np.__version__, solver='Glucose42', seed=args.seed,
                    repeats=args.repeats, timeout_seconds=args.timeout,
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    maximal_candidate=args.maximal, encodings=args.encodings,
                    cases_path=str(cases_path.resolve()), source_hashes=source_hashes,
                    total_generated=len(cases), total_reaching_sat=sum(c['branch']=='sat' for c in cases),
                    selected_case_ids=[c['case_id'] for c in selected])
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    output = args.out / 'runs.jsonl'
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a fresh --out directory')
    print(json.dumps({k:metadata[k] for k in ['total_generated','total_reaching_sat','encodings','repeats','timeout_seconds']}), flush=True)
    known = {}
    # Warm solver/library initialization on a trivial disposable formula.
    with Glucose42(bootstrap_with=[[1]]) as warm:
        assert warm.solve()
    with output.open('w') as fp:
        for ci, case in enumerate(selected):
            initial_order = list(args.encodings)
            random.Random(args.seed + int(case['case_id'][1:])).shuffle(initial_order)
            for repeat in range(args.repeats):
                # Rotate order: with three encodings/repeats, each occupies each position.
                order = initial_order[repeat % len(initial_order):] + initial_order[:repeat % len(initial_order)]
                for position, encoding in enumerate(order):
                    result = run_one(case, encoding, args.maximal, args.timeout)
                    result.update(repeat=repeat, order_position=position)
                    if result['status'] != 'UNKNOWN':
                        old = known.setdefault(case['case_id'], result['status'])
                        if old != result['status']:
                            raise RuntimeError(f'Disagreement for {case["case_id"]}')
                    fp.write(json.dumps(result) + '\n')
                    fp.flush()
                    print(json.dumps({k:result[k] for k in ['case_id','cell','encoding','repeat','status','solve_seconds','sat_stage_seconds']}), flush=True)
    print(json.dumps(dict(completed=True, cases=len(selected), runs=len(selected)*args.repeats*len(args.encodings))), flush=True)


if __name__ == '__main__':
    main()
