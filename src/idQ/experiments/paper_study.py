"""Prepare and run the 36 settings in the revised idQ simulation study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np


def settings():
    cells = []
    for J in (25, 50, 100):
        for K in (5, 10):
            for p in (0.1, 0.3, 0.5, 0.7, 0.9):
                cells.append(dict(cell_id=f"bern_J{J}_K{K}_p{p:.1f}",
                                  design="bernoulli", J=J, K=K, p=p,
                                  zero_row_policy="allow_iid"))
    for J in (25, 50, 100):
        for m in (3, 4):
            cells.append(dict(cell_id=f"sparse_J{J}_K10_m{m}",
                              design="row_sparsity", J=J, K=10,
                              m_requested=m, m_effective=m,
                              min_row_size_requested=1,
                              row_size_distribution="uniform"))
    return cells


def load_manifest(study_dir):
    root = Path(study_dir).expanduser().resolve()
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        return root, json.load(handle)


def prepare(study_dir, *, replicates=1000, per_task=100, base_seed=20260922):
    from .common import source_provenance
    if replicates <= 0 or per_task <= 0 or base_seed < 0:
        raise ValueError("replicates/per-task must be positive and base-seed nonnegative")
    root = Path(study_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for cell_index, cell in enumerate(settings()):
        for batch, start in enumerate(range(0, replicates, per_task)):
            seed = int(np.random.SeedSequence([base_seed, cell_index, batch])
                       .generate_state(1)[0])
            relative = f"raw/{cell['cell_id']}/batch_{batch:03d}.csv"
            tasks.append(dict(cell, task_id=len(tasks), batch=batch,
                              N=min(per_task, replicates-start), seed=seed,
                              csv_path=relative, metadata_path=relative+".metadata.json"))
    source = source_provenance()
    manifest = dict(manifest_version=1, study_id=root.name,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    expected_replicates_per_cell=replicates,
                    replicates_per_task=per_task, base_seed=base_seed,
                    solver_name="glucose42", cardinality_encoding="exclude_x",
                    maximal_candidate=False, rng_engine="MT19937-RandomState",
                    source_tree_sha256=source["source_tree_sha256"],
                    source_hashes=source["source_hashes"], tasks=tasks)
    # An existing study is never silently redefined.
    with (root / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (root / "logs").mkdir(exist_ok=True)
    print(f"Prepared {len(tasks)} tasks, {len(tasks) and len(settings())} settings, "
          f"{replicates * len(settings()):,} matrices: {root}")
    return manifest


def check_source(manifest):
    from .common import source_provenance
    actual = source_provenance()["source_tree_sha256"]
    if actual != manifest["source_tree_sha256"]:
        raise RuntimeError("Package source changed since study preparation. "
                           "Use the original code or prepare a new study directory.")


def task_state(root, task):
    csv_path = root / task["csv_path"]
    metadata = root / task["metadata_path"]
    if csv_path.exists() and metadata.exists():
        return "complete"  # The analysis command validates the contents in full.
    if any(p.exists() for p in (csv_path, metadata,
                               Path(str(csv_path)+".part"),
                               Path(str(metadata)+".part"))):
        return "incomplete"
    return "missing"


def array_spec(indices):
    """Compact a sorted collection of task IDs into Slurm array ranges."""
    chunks = []
    values = sorted(set(indices))
    start = end = None
    for value in values:
        if start is None:
            start = end = value
        elif value == end + 1:
            end = value
        else:
            chunks.append(str(start) if start == end else f"{start}-{end}")
            start = end = value
    if start is not None:
        chunks.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(chunks)


def run_task(study_dir, task_id, *, retry_incomplete=False):
    from .bernoulli import run_expr as run_bernoulli
    from .row_sparsity import run_expr as run_sparse
    root, manifest = load_manifest(study_dir)
    check_source(manifest)
    if not 0 <= task_id < len(manifest["tasks"]):
        raise ValueError(f"task-id must be in 0..{len(manifest['tasks'])-1}")
    task = manifest["tasks"][task_id]
    output = root / task["csv_path"]
    metadata = root / task["metadata_path"]
    state = task_state(root, task)
    if state == "complete":
        print(f"Task {task_id} already has completed output; left unchanged.")
        return
    if state == "incomplete":
        if not retry_incomplete:
            raise FileExistsError(f"Task {task_id} has partial output. After ensuring "
                                  "it is no longer running, use --retry-incomplete.")
        archive = root / "attempts" / (f"task_{task_id}_" +
                   datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        archive.mkdir(parents=True)
        for path in (output, metadata, Path(str(output)+".part"),
                     Path(str(metadata)+".part")):
            if path.exists():
                shutil.move(str(path), archive / path.name)
        print(f"Archived partial task {task_id} to {archive}")
    common = dict(J=task["J"], K=task["K"], N=task["N"], seed=task["seed"],
                  solver="glucose42", cardinality_encoding="exclude_x",
                  maximal_candidate=False, rng_engine="legacy",
                  output_csv=str(output), checkpoint_every=10)
    print(f"Task {task_id}: {task['cell_id']}, N={task['N']}, seed={task['seed']}",
          flush=True)
    if task["design"] == "bernoulli":
        run_bernoulli(**common, p=task["p"], condition_nonzero_rows=False)
    else:
        run_sparse(**common, m=task["m_requested"], min_row_size=1,
                   row_size_distribution="uniform")
    print(f"Completed task {task_id}: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--study-dir", type=Path, required=True)
    prep.add_argument("--replicates", type=int, default=1000)
    prep.add_argument("--per-task", type=int, default=100)
    prep.add_argument("--base-seed", type=int, default=20260922)
    run = sub.add_parser("run-task")
    run.add_argument("--study-dir", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--retry-incomplete", action="store_true")
    indices = sub.add_parser("task-indices")
    indices.add_argument("--study-dir", type=Path, required=True)
    indices.add_argument("--missing", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.study_dir, replicates=args.replicates, per_task=args.per_task,
                base_seed=args.base_seed)
    elif args.command == "run-task":
        run_task(args.study_dir, args.task_id, retry_incomplete=args.retry_incomplete)
    else:
        root, manifest = load_manifest(args.study_dir)
        check_source(manifest)
        states = [(task["task_id"], task_state(root, task)) for task in manifest["tasks"]]
        if args.command == "task-indices":
            print(array_spec(i for i, state in states
                             if not args.missing or state != "complete"))
        else:
            counts = {state: sum(s == state for _, s in states)
                      for state in ("complete", "incomplete", "missing")}
            print(json.dumps(dict(total_tasks=len(states), **counts), indent=2))
            print("Completion here checks file presence; paper_summary validates contents.")


if __name__ == "__main__":
    main()
