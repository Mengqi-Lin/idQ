#!/usr/bin/env python3
"""Summarize paired benchmark runs without treating UNKNOWN as UNSAT."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import hashlib
import math
from pathlib import Path
import statistics as st


def median(values):
    return st.median(values) if values else None


def analyze(path):
    metadata = json.loads((path / 'metadata.json').read_text())
    rows = [json.loads(line) for line in (path / 'runs.jsonl').read_text().splitlines()]
    by_pair = defaultdict(list)
    statuses = defaultdict(set)
    for r in rows:
        by_pair[r['case_id'], r['encoding']].append(r)
        if r['status'] != 'UNKNOWN':
            statuses[r['case_id']].add(r['status'])
    assert all(len(x) <= 1 for x in statuses.values()), 'Contradictory SAT decisions'
    expected_pairs = {(c,e) for c in metadata['selected_case_ids'] for e in metadata['encodings']}
    assert set(by_pair) == expected_pairs, 'Run is incomplete: missing case/encoding groups'
    for key, group in by_pair.items():
        assert sorted(r['repeat'] for r in group) == list(range(metadata['repeats'])), f'Incomplete/duplicate repetitions: {key}'
    original_cases = len(metadata['selected_case_ids'])
    corpus_path = Path(metadata['cases_path'])
    if not corpus_path.exists():
        corpus_path = path.parent / corpus_path.name
    corpus = json.loads(corpus_path.read_text())['cases']
    representatives = {}
    mapping = {}
    members = defaultdict(list)
    for case in corpus:
        if case['case_id'] not in metadata['selected_case_ids']:
            continue
        key = hashlib.sha256(json.dumps([case['K'], case['basis']]).encode()).hexdigest()
        representative = representatives.setdefault(key, case['case_id'])
        mapping[case['case_id']] = representative
        members[representative].append(case['case_id'])
    # Pool repeated timings of exactly identical SAT inputs; count each basis once.
    by_pair = defaultdict(list)
    statuses = defaultdict(set)
    for r in rows:
        r['original_case_id'] = r['case_id']
        r['case_id'] = mapping[r['case_id']]
        by_pair[r['case_id'], r['encoding']].append(r)
        if r['status'] != 'UNKNOWN':
            statuses[r['case_id']].add(r['status'])
    assert all(len(x) <= 1 for x in statuses.values()), 'Contradictory decisions for duplicate bases'
    encodings = sorted({r['encoding'] for r in rows}, key=lambda e:(e!='prefix', e))
    cases = sorted({r['case_id'] for r in rows})
    aggregates = {}
    for enc in encodings:
        records = []
        for cid in cases:
            group = by_pair.get((cid, enc), [])
            if not group:
                continue
            all_solved = all(r['status'] != 'UNKNOWN' for r in group)
            rec = dict(case_id=cid, encoding=enc, family=group[0]['family'], cell=group[0]['cell'],
                       equivalent_input_ids=members[cid],
                       status=next(iter(statuses[cid]), 'UNKNOWN'), all_repeats_solved=all_solved,
                       timeout_runs=sum(r['status']=='UNKNOWN' for r in group), repeats=len(group),
                       variables=group[0]['variables'], clauses=group[0]['clauses'],
                       literals=group[0]['literals'], Jb=group[0]['Jb'])
            for key in ('build_seconds','load_seconds','solve_seconds','sat_stage_seconds', 'solve_cpu_seconds'):
                rec[key] = median([r[key] for r in group])
            records.append(rec)
        aggregates[enc] = records

    summary = dict(cases=len(cases), original_sat_stage_inputs=original_cases, runs=len(rows),
                   case_statuses=dict(Counter(next(iter(statuses[cid]),'UNKNOWN') for cid in cases)),
                   validated_sat_runs=sum(r['validated'] is True for r in rows), encodings={})
    baseline = {r['case_id']:r for r in aggregates.get('prefix', [])}
    common = set.intersection(*[{r['case_id'] for r in records if r['all_repeats_solved']}
                               for records in aggregates.values()])
    summary['common_complete_cases'] = len(common)
    for enc, records in aggregates.items():
        solved = [r for r in records if r['all_repeats_solved']]
        paired = [r for r in solved if r['case_id'] in baseline and baseline[r['case_id']]['all_repeats_solved']]
        ratios = [baseline[r['case_id']]['sat_stage_seconds']/r['sat_stage_seconds'] for r in paired]
        summary['encodings'][enc] = dict(
            complete_cases=len(solved), timeout_cases=sum(r['timeout_runs']>0 for r in records),
            timeout_runs=sum(r['timeout_runs'] for r in records),
            median_solve_ms=1000*median([r['solve_seconds'] for r in solved]) if solved else None,
            median_stage_ms=1000*median([r['sat_stage_seconds'] for r in solved]) if solved else None,
            total_stage_seconds=sum(r['sat_stage_seconds'] for r in solved),
            total_solve_seconds=sum(r['solve_seconds'] for r in solved),
            max_solve_seconds=max((r['solve_seconds'] for r in solved),default=None),
            paired_complete_cases=len(paired), paired_geomean_stage_speedup=math.exp(st.mean(math.log(x) for x in ratios)) if ratios else None,
            paired_aggregate_stage_speedup=sum(baseline[r['case_id']]['sat_stage_seconds'] for r in paired)/sum(r['sat_stage_seconds'] for r in paired) if paired else None,
            faster_stage_cases=sum(x>1 for x in ratios),
            median_build_ms=1000*median([r['build_seconds'] for r in solved]) if solved else None,
            median_load_ms=1000*median([r['load_seconds'] for r in solved]) if solved else None,
            median_variables=median([r['variables'] for r in records]),
            median_clauses=median([r['clauses'] for r in records]),
            common_complete_stage_seconds=sum(r['sat_stage_seconds'] for r in records if r['case_id'] in common),
            common_complete_solve_seconds=sum(r['solve_seconds'] for r in records if r['case_id'] in common),
            # Median per-instance PAR2 across repeats; censored solves cost 2*limit.
            par2_solve_score=sum(median([2*r['timeout_seconds'] if r['status']=='UNKNOWN' else r['solve_seconds']
                                         for r in by_pair[(cid,enc)]]) for cid in cases),
        )
    breakdown=[]
    for label in ('SAT','UNSAT','bernoulli','fixed_weight','mixed_sparse','structured','planted_sat'):
        for enc, records in aggregates.items():
            sub=[r for r in records if (r['status']==label or r['family']==label) and r['all_repeats_solved']]
            if sub:
                breakdown.append(dict(group=label,encoding=enc,cases=len(sub),
                                      total_solve_seconds=sum(r['solve_seconds'] for r in sub),
                                      total_stage_seconds=sum(r['sat_stage_seconds'] for r in sub),
                                      median_solve_ms=1000*median([r['solve_seconds'] for r in sub])))
    summary['breakdown']=breakdown
    (path/'summary.json').write_text(json.dumps(summary,indent=2))
    (path/'case_medians.json').write_text(json.dumps(aggregates,indent=2))
    lines=['# Single-clause exclusion benchmark', '',
           f'{original_cases} SAT-stage inputs, {len(cases)} distinct basis matrices; {len(rows)} timed runs. Distinct-basis outcomes: {summary["case_statuses"]}.', '',
           'Timings for exactly identical basis matrices are pooled. Each distinct basis is summarized by the median of its repetitions. Total times sum those medians; they are not elapsed experiment time.',
           'The SAT-stage time includes CNF construction, solver initialization, and solve time. Sampling, shared preprocessing, validation, timer management, garbage collection, and solver destruction are excluded.',
           'UNKNOWN runs are counted as timeouts, never as UNSAT. Speedups use only cases where every repeat completed for both compared encodings.', '',
           '| Encoding | Complete cases | Timeout cases | Median solve (ms) | Total solve (s) | Total SAT stage (s) | Paired aggregate speedup |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.3f}'
    for enc,a in summary['encodings'].items():
        lines.append(f'| {enc} | {a["complete_cases"]} | {a["timeout_cases"]} | {fmt(a["median_solve_ms"])} | {fmt(a["total_solve_seconds"])} | {fmt(a["total_stage_seconds"])} | {fmt(a["paired_aggregate_stage_speedup"])} |')
    lines += ['', '| Group | Encoding | Cases | Total solve (s) | Total SAT stage (s) |', '|---|---|---:|---:|---:|']
    for r in breakdown:
        lines.append(f'| {r["group"]} | {r["encoding"]} | {r["cases"]} | {r["total_solve_seconds"]:.3f} | {r["total_stage_seconds"]:.3f} |')
    lines += ['', 'Environment and source hashes are in metadata.json. Exact inputs are in the corpus cases.json. Every SAT witness is in runs.jsonl and was independently checked using Boolean multiplication and column-permutation comparison.', '']
    (path/'SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path)
    analyze(p.parse_args().path)
