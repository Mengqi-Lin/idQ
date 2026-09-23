"""Validate a manifest-backed paper study and regenerate its numerical tables.

Completed files are accepted only with matching completed provenance sidecars.
The partial mode skips missing/interrupted tasks; it never relaxes validation of
files that are included, and never writes a paper-final table.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..core import BRANCH_LABELS
from .paper_study import settings as study_settings


JS = (25, 50, 100)
KS = (5, 10)
PS = (.1, .3, .5, .7, .9)
MS = (3, 4)
SETTINGS = dict(solver_name="glucose42", cardinality_encoding_requested="exclude_x",
                maximal_candidate_requested=False, rng_engine="MT19937-RandomState")
TIME_FIELDS = ("basis_time", "identity_check_time", "two_col_check_time",
               "three_col_check_time", "sat_time", "preprocess_time", "algorithm_time")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected an integer, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name}: expected a finite integer")
    return int(number)


def _binary(value: Any, name: str) -> int:
    result = int(value) if isinstance(value, bool) else _integer(value, name)
    if result not in (0, 1):
        raise ValueError(f"{name}: expected 0 or 1")
    return result


def _cell_key(config: dict) -> tuple:
    design = config.get("design")
    J, K = _integer(config.get("J"), "J"), _integer(config.get("K"), "K")
    if design == "bernoulli":
        return design, J, K, round(float(config["p"]), 12)
    if design == "row_sparsity":
        return design, J, K, _integer(config.get("m_requested"), "m_requested")
    raise ValueError(f"Unexpected design {design!r}")


def expected_grid(profile="paper") -> set[tuple]:
    return {_cell_key(cell) for cell in study_settings(profile)}


def _relative(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Study paths must be relative and remain inside the study: {name}")
    return path


def _settings(config: dict, context: str, *, manifest: bool = False) -> None:
    aliases = {"cardinality_encoding_requested": "cardinality_encoding",
               "maximal_candidate_requested": "maximal_candidate"} if manifest else {}
    for key, expected in SETTINGS.items():
        value = config.get(key, config.get(aliases.get(key, key)))
        if key == "maximal_candidate_requested":
            value = bool(_binary(value, f"{context}: {key}"))
        if value != expected:
            raise ValueError(f"{context}: {key} must be {expected!r}, found {value!r}")


def _design(config: dict, context: str, profile="paper") -> None:
    key = _cell_key(config)
    if key not in expected_grid(profile):
        raise ValueError(f"{context}: design is outside the declared paper grid: {key}")
    if key[0] == "bernoulli":
        if config.get("zero_row_policy") != "allow_iid":
            raise ValueError(f"{context}: Bernoulli must use unconditioned iid rows (allow_iid)")
    else:
        for name, expected in (("m_effective", key[3]), ("min_row_size_requested", 1)):
            if _integer(config.get(name), f"{context}: {name}") != expected:
                raise ValueError(f"{context}: incorrect {name}")
        if config.get("row_size_distribution") != "uniform":
            raise ValueError(f"{context}: row sizes must be uniform on 1,...,m")


def validate_manifest(manifest: dict, root: Path) -> list[dict]:
    if manifest.get("manifest_version") != 1:
        raise ValueError("Expected manifest_version=1")
    _settings(manifest, "manifest", manifest=True)
    profile = manifest.get("study_profile", "paper")
    grid = expected_grid(profile)
    expected = _integer(manifest.get("expected_replicates_per_cell"), "expected_replicates_per_cell")
    if expected <= 0:
        raise ValueError("expected_replicates_per_cell must be positive")
    source = manifest.get("source_tree_sha256", "")
    if len(source) != 64 or any(x not in "0123456789abcdef" for x in source):
        raise ValueError("Manifest must record source_tree_sha256")
    tasks = manifest.get("tasks", [])
    if not tasks:
        raise ValueError("Manifest has no tasks")
    ids, paths, seeds, by_cell, cell_ids = set(), set(), set(), {}, {}
    for task in tasks:
        _design(task, "manifest task", profile)
        key = _cell_key(task)
        task_id = str(task["task_id"])
        if task_id in ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        ids.add(task_id)
        N = _integer(task["N"], "task N")
        seed = _integer(task["seed"], "task seed")
        if N <= 0 or seed < 0 or (key, seed) in seeds:
            raise ValueError(f"Invalid N/seed or duplicate cell/seed: {task_id}")
        seeds.add((key, seed))
        cid = str(task["cell_id"])
        if cid in cell_ids and cell_ids[cid] != key:
            raise ValueError(f"cell_id reused for different designs: {cid}")
        cell_ids[cid] = key
        for field in ("csv_path", "metadata_path"):
            path = _relative(root, task[field])
            if path in paths:
                raise ValueError(f"Duplicate manifest output path: {path}")
            paths.add(path)
        by_cell[key] = by_cell.get(key, 0) + N
    if set(by_cell) != grid:
        if profile == "paper":
            raise ValueError("Manifest must contain exactly the 30 Bernoulli and 6 sparsity cells")
        raise ValueError("Manifest must contain exactly the 30 K=20,30 Bernoulli cells")
    if any(count != expected for count in by_cell.values()):
        raise ValueError("Manifest task counts do not match expected_replicates_per_cell")
    return tasks


def _sequence(value: Any, cast=float) -> list:
    if isinstance(value, str):
        value = value.split(";")
    return [cast(x) for x in value]


def _validate_row(row: dict, task: dict, manifest: dict, context: str) -> dict:
    _settings(row, context)
    _design(row, context, manifest.get("study_profile", "paper"))
    if manifest.get("study_profile") == "bernoulli_large":
        if _binary(row.get("M_basis_computed"), "M_basis_computed") or row.get("M_basis") not in (None, ""):
            raise ValueError(f"{context}: the large-K profile must skip class counting")
    if _cell_key(row) != _cell_key(task):
        raise ValueError(f"{context}: wrong design cell")
    if row.get("operator") != "conj" or _integer(row.get("schema_version"), "schema_version") != 4:
        raise ValueError(f"{context}: expected conjunctive schema-v4 data")
    for name in ("N", "seed", "J", "K"):
        if _integer(row.get(name), f"{context}: {name}") != _integer(task[name], name):
            raise ValueError(f"{context}: wrong {name}")
    if not row.get("design_id"):
        raise ValueError(f"{context}: missing design_id")
    if row["design"] == "row_sparsity":
        m = _cell_key(task)[3]
        if _sequence(row.get("row_size_support", ""), int) != list(range(1, m + 1)):
            raise ValueError(f"{context}: wrong row-size support")
        probs = _sequence(row.get("row_size_probs", ""))
        if len(probs) != m or not np.allclose(probs, np.full(m, 1 / m), rtol=0, atol=1e-12):
            raise ValueError(f"{context}: row sizes are not uniform")
    result = dict(row)
    for name in ("sim", "N", "seed", "J", "K", "J_basis", "branch"):
        result[name] = _integer(row.get(name), f"{context}: {name}")
    for name in ("identifiable", "sat_called", "identity_original", "not_complete_original",
                 "has_pure_node_original", "no_pure_nodes_original", "has_zero_row_original",
                 "two_column_pass_original", "both_necessary_checks_pass_original", "I_3c",
                 "two_col_violation_original", "three_col_violation_original"):
        result[name] = _binary(row.get(name), f"{context}: {name}")
    for name in TIME_FIELDS:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context}: missing/nonnumeric {name}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{context}: {name} must be finite and nonnegative")
        result[name] = value
    if not math.isclose(result["preprocess_time"], sum(result[n] for n in TIME_FIELDS[:4]), rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"{context}: inconsistent preprocessing time")
    if result["algorithm_time"] + 1e-9 < result["preprocess_time"] + result["sat_time"]:
        raise ValueError(f"{context}: algorithm time is shorter than its measured stages")
    branch = result["branch"]
    if branch not in BRANCH_LABELS or row.get("branch_label") != BRANCH_LABELS[branch]:
        raise ValueError(f"{context}: invalid branch/label")
    if result["identifiable"] != int(branch in (4, 6)) or result["sat_called"] != int(branch in (5, 6)):
        raise ValueError(f"{context}: inconsistent branch outcome")
    sat = result["sat_called"]
    if sat:
        if row.get("sat_cardinality_encoding") != "exclude_x" or row.get("sat_formulation") != "boolean_factorization":
            raise ValueError(f"{context}: SAT result used the wrong formulation or encoding")
    elif row.get("sat_cardinality_encoding", "") or result["sat_time"] != 0:
        raise ValueError(f"{context}: unexpected SAT data for a preprocessing branch")
    J, K = result["J"], result["K"]
    bits = row.get("q_bitstring", "")
    if len(bits) != J * K or set(bits) - {"0", "1"}:
        raise ValueError(f"{context}: invalid q_bitstring")
    digest = hashlib.sha256(f"{J},{K}:{bits}".encode("ascii")).hexdigest()
    if row.get("q_sha256") != digest:
        raise ValueError(f"{context}: Q fingerprint mismatch")
    Q = np.fromiter((int(c) for c in bits), dtype=np.int64).reshape(J, K)
    row_sizes = Q.sum(axis=1)
    complete = bool(np.all(np.any(Q[row_sizes == 1], axis=0)))
    # Counts of (1,0) and (0,1) for every column pair.
    col_sizes = Q.sum(axis=0)
    intersections = Q.T @ Q
    two_pass = bool(np.all((col_sizes[:, None] - intersections + np.eye(K, dtype=int)) > 0))
    triples = np.asarray(list(itertools.combinations(range(K), 3)))
    triple_pass = bool(np.all(np.any(Q[:, triples].sum(axis=2) == 1, axis=0)))
    checks = {"identity_original": int(complete), "not_complete_original": int(not complete),
              "has_pure_node_original": int(np.any(row_sizes == 1)),
              "no_pure_nodes_original": int(not np.any(row_sizes == 1)),
              "has_zero_row_original": int(np.any(row_sizes == 0)),
              "two_column_pass_original": int(two_pass), "two_col_violation_original": int(not two_pass),
              "three_col_violation_original": int(not triple_pass),
              "both_necessary_checks_pass_original": int(two_pass and triple_pass),
              "I_3c": int(two_pass and triple_pass)}
    if any(result[name] != value for name, value in checks.items()):
        raise ValueError(f"{context}: original-Q indicators do not match the recorded matrix")
    if row["design"] == "row_sparsity" and (row_sizes.min() < 1 or row_sizes.max() > _cell_key(task)[3]):
        raise ValueError(f"{context}: matrix violates the sparsity design")
    if not 0 <= result["J_basis"] <= J:
        raise ValueError(f"{context}: impossible basis row count")
    if not (result["identity_original"] <= result["identifiable"] <= result["I_3c"] <= result["two_column_pass_original"]):
        raise ValueError(f"{context}: indicator hierarchy fails")
    if sat != result["I_3c"] - result["identity_original"]:
        raise ValueError(f"{context}: SAT invocation disagrees with the preprocessing checks")
    return result


def read_study(study_dir: Path, *, allow_partial: bool = False) -> tuple[dict, list[dict], list[str]]:
    root = Path(study_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = validate_manifest(manifest, root)
    expected_files = {_relative(root, t["csv_path"]) for t in tasks}
    raw = root / "raw"
    extras = set(raw.rglob("*.csv")) - expected_files if raw.exists() else set()
    if extras:
        raise ValueError(f"Unlisted CSV files in study: {sorted(map(str, extras))}")
    issues = [f"Incomplete file: {p.relative_to(root)}" for p in sorted(raw.rglob("*.part"))] if raw.exists() else []
    data: list[dict] = []
    environments = []
    software_reference = None
    timing_reference = None
    for task in tasks:
        csv_path, metadata_path = (_relative(root, task[k]) for k in ("csv_path", "metadata_path"))
        if not csv_path.exists() or not metadata_path.exists():
            issues.append(f"Missing completed task {task['task_id']}: {task['csv_path']} or its metadata")
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != 4:
            raise ValueError(f"{metadata_path}: expected schema-v4 provenance")
        if metadata.get("status") != "completed" or metadata.get("completed") is not True:
            issues.append(f"Uncompleted metadata for task {task['task_id']}")
            continue
        _settings(metadata, str(metadata_path))
        _design(metadata, str(metadata_path), manifest.get("study_profile", "paper"))
        if manifest.get("study_profile") == "bernoulli_large" and metadata.get("compute_class_count") is not False:
            raise ValueError(f"{metadata_path}: the large-K profile must skip class counting")
        if _cell_key(metadata) != _cell_key(task):
            raise ValueError(f"{metadata_path}: wrong cell")
        for key in ("J", "K", "N", "seed"):
            if _integer(metadata.get(key), key) != _integer(task[key], key):
                raise ValueError(f"{metadata_path}: {key} disagrees with manifest")
        hashes = metadata.get("source_hashes", {})
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if not hashes or digest != manifest["source_tree_sha256"] or metadata.get("source_tree_sha256") != digest:
            raise ValueError(f"{metadata_path}: source provenance differs from the study manifest")
        if not metadata.get("environment") or not metadata.get("timing_definitions"):
            raise ValueError(f"{metadata_path}: missing environment/timing provenance")
        environment = metadata["environment"]
        software = {key: environment.get(key) for key in
                    ("python_version", "pysat_version", "numpy_version", "idQ_version")}
        if not all(isinstance(value, str) and value for value in software.values()):
            raise ValueError(f"{metadata_path}: missing software-version provenance")
        if software_reference is not None and software != software_reference:
            raise ValueError(f"{metadata_path}: mixed software versions across study jobs")
        if timing_reference is not None and metadata["timing_definitions"] != timing_reference:
            raise ValueError(f"{metadata_path}: mixed timing definitions across study jobs")
        software_reference = software
        timing_reference = metadata["timing_definitions"]
        environments.append(dict(task_id=task["task_id"], N=task["N"], environment=environment))
        with csv_path.open(newline="") as fp:
            records = list(csv.DictReader(fp))
        N = _integer(task["N"], "N")
        if len(records) != N or _integer(metadata.get("completed_replicates"), "completed_replicates") != N:
            raise ValueError(f"{csv_path}: completed task has missing or duplicated records")
        validated = [_validate_row(r, task, manifest, f"{csv_path}: row {i + 2}") for i, r in enumerate(records)]
        if sorted(r["sim"] for r in validated) != list(range(N)):
            raise ValueError(f"{csv_path}: missing or duplicate simulation indices")
        if {r["design_id"] for r in validated} != {metadata.get("design_id")}:
            raise ValueError(f"{csv_path}: design_id differs from metadata")
        for record in validated:
            record.update(cell_id=task["cell_id"], task_id=task["task_id"], source_file=task["csv_path"])
        data.extend(validated)
    if issues and not allow_partial:
        raise ValueError("Study is incomplete; final summaries are refused. " + "; ".join(issues[:8])
                         + ". Use --allow-partial for a clearly marked interim summary.")
    def counts(function):
        return dict(Counter(str(function(entry["environment"])) for entry in environments))
    models = counts(lambda e: e.get("cpu_model") or "unknown")
    manifest = dict(manifest, _validated_runtime=dict(
        completed_jobs=len(environments), software_versions=software_reference,
        timing_definitions=timing_reference,
        cpu_model_job_counts=models,
        mixed_hardware=len(models) > 1,
        hostname_job_counts=counts(lambda e: e.get("hostname") or "unknown"),
        execution_mode_job_counts=counts(lambda e: "slurm" if e.get("slurm", {}).get("SLURM_JOB_ID") or e.get("slurm", {}).get("SLURM_ARRAY_JOB_ID") else "local"),
        slurm_cpus_per_task_job_counts=counts(lambda e: e.get("slurm", {}).get("SLURM_CPUS_PER_TASK", "not_recorded")),
        cpu_affinity_size_job_counts=counts(lambda e: len(e["cpu_affinity"]) if isinstance(e.get("cpu_affinity"), list) else "not_recorded"),
        thread_environment_job_counts=counts(lambda e: json.dumps(e.get("thread_environment", {}), sort_keys=True)),
        jobs=environments,
    ))
    return manifest, data, issues


def summarize_cells(manifest: dict, data: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {key: [] for key in expected_grid(manifest.get("study_profile", "paper"))}
    for row in data:
        grouped[_cell_key(row)].append(row)
    result = []
    fields = {"identifiable": "identifiable", "complete": "identity_original", "two_column": "two_column_pass_original",
              "both_necessary": "I_3c", "no_pure_nodes": "no_pure_nodes_original", "sat_called": "sat_called"}
    for key, rows in sorted(grouped.items()):
        n = len(rows)
        rec = dict(design=key[0], J=key[1], K=key[2], p=key[3] if key[0] == "bernoulli" else None,
                   m=key[3] if key[0] == "row_sparsity" else None, n=n,
                   expected_n=manifest["expected_replicates_per_cell"])
        for label, field in fields.items():
            count = sum(r[field] for r in rows)
            rec["n_" + label] = count
            rec["pct_" + label] = 100 * count / n if n else None
        denominator = rec["n_identifiable"]
        rec["conditional_denominator_identifiable"] = denominator
        for label, field in (("incomplete_given_identifiable", "not_complete_original"),
                             ("no_pure_given_identifiable", "no_pure_nodes_original")):
            count = sum(r["identifiable"] * r[field] for r in rows)
            rec["n_" + label] = count
            rec["pct_" + label] = 100 * count / denominator if denominator else None
        for field in ("J_basis",) + TIME_FIELDS:
            total = sum(r[field] for r in rows)
            rec["sum_" + field] = total
            rec["mean_" + field] = total / n if n else None
        rec["branch_counts"] = {str(b): sum(r["branch"] == b for r in rows) for b in BRANCH_LABELS}
        result.append(rec)
    return result


def _fmt(value: float | None, *, runtime=False) -> str:
    if value is None:
        return "--"
    if runtime:
        return r"\(<0.001\)" if 0 < value < .001 else f"{value:.3f}"
    return f"{value:.1f}"


def latex_tables(cells: list[dict], status: str, profile="paper") -> str:
    ks = (5, 10) if profile == "paper" else (20, 30)
    label_suffix = "" if profile == "paper" else "-large"
    lookup = {(r["design"], r["J"], r["K"], r["p"] if r["design"] == "bernoulli" else r["m"]): r for r in cells}
    warning = "" if status in ("paper_complete", "extension_complete") else (r"\textbf{PARTIAL RESULTS.} " if status == "partial" else r"\textbf{SMOKE STUDY; NOT PAPER RESULTS.} ")
    captions = {
        "pct_identifiable": r"\(\widehat{\P}(I_{\mathrm{id}}=1)\).",
        "pct_complete": r"\(\widehat{\P}(I_{\mathrm{comp}}=1)\).",
        "pct_two_column": r"\(\widehat{\P}(I_{\mathrm{2c}}=1)\).",
        "pct_both_necessary": r"\(\widehat{\P}(I_{\mathrm{3c}}=1)\).",
        "pct_incomplete_given_identifiable": r"\(\widehat{\P}(I_{\mathrm{comp}}=0\mid I_{\mathrm{id}}=1)\).",
        "pct_no_pure_given_identifiable": r"\(\widehat{\P}(I_{\mathrm{np}}=1\mid I_{\mathrm{id}}=1)\)."}

    def panel(metric, sparse=False):
        lines = [r"\begin{subtable}[t]{0.48\textwidth}", r"\centering"]
        if sparse:
            lines += [r"\begin{tabular}{c cc}", r"\toprule", r"\(J\) & \(m=3\) & \(m=4\)\\", r"\midrule"]
            for J in JS:
                lines.append(f"{J} & " + " & ".join(_fmt(lookup['row_sparsity', J, 10, m][metric]) for m in MS) + r"\\")
        else:
            lines += [r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{c cc cc cc}", r"\toprule",
                      r"& \multicolumn{2}{c}{\(J=25\)} & \multicolumn{2}{c}{\(J=50\)} & \multicolumn{2}{c}{\(J=100\)}\\",
                      r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                      r"\(p\) & " + " & ".join(r"\(K=" + str(K) + r"\)" for J in JS for K in ks) + r"\\", r"\midrule"]
            for p in PS:
                lines.append(f"{p:.1f} & " + " & ".join(_fmt(lookup['bernoulli', J, K, p][metric]) for J in JS for K in ks) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        if not sparse:
            lines += ["}"]
        lines += [r"\caption{" + captions[metric] + "}"]
        sublabels = {"pct_identifiable": "tab:bern-id", "pct_two_column": "tab:bern-2c",
                     "pct_both_necessary": "tab:bern-3c", "pct_no_pure_given_identifiable": "tab:bern-np-given-id"}
        if not sparse and metric in sublabels:
            lines.append(r"\label{" + sublabels[metric] + label_suffix + "}")
        lines.append(r"\end{subtable}")
        return "\n".join(lines)

    tables = [f"% Automatically generated; status={status}. Requires booktabs, subcaption, graphicx."]
    specs = [("tab:bern", ["pct_identifiable", "pct_complete", "pct_two_column", "pct_both_necessary"], False,
              "Empirical proportions under the Bernoulli design, expressed as percentages."),
             ("tab:bern-pure-node-gap", ["pct_incomplete_given_identifiable", "pct_no_pure_given_identifiable"], False,
              "Conditional empirical proportions among identifiable matrices under the Bernoulli design, expressed as percentages. A dash indicates a zero identifiable denominator or a missing cell in partial results."),
             ("tab:sparse", ["pct_identifiable", "pct_incomplete_given_identifiable", "pct_two_column", "pct_both_necessary"], True,
              "Empirical proportions under the sparsity design, expressed as percentages.")]
    for label, metrics, sparse, caption in specs:
        if sparse and profile != "paper":
            continue
        lines = [r"\begin{table}[htbp]", r"\centering"]
        for i, metric in enumerate(metrics):
            if i:
                lines.append(r"\hfill" if i % 2 else "\n" + r"\vspace{0.8em}" + "\n")
            lines.append(panel(metric, sparse))
        lines += [r"\caption{" + warning + caption + "}", r"\label{" + label + label_suffix + "}", r"\end{table}"]
        tables.append("\n".join(lines))
    for K in ((10,) if profile == "paper" else ks):
        lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{ccrrrr}", r"\toprule",
                 r"\(J\) & \(p\) & Mean \(J_b\) & SAT invoked (\%) & Steps~0--1 runtime & Total runtime\\", r"\midrule"]
        for j, J in enumerate(JS):
            if j:
                lines.append(r"\addlinespace")
            for p in PS:
                r = lookup['bernoulli', J, K, p]
                values = [_fmt(r['mean_J_basis']), _fmt(r['pct_sat_called']),
                          _fmt(r['mean_preprocess_time'], runtime=True), _fmt(r['mean_algorithm_time'], runtime=True)]
                lines.append(f"{J} & {p:.1f} & " + " & ".join(values) + r"\\")
        label = "tab:bern-computation" + ("" if profile == "paper" else f"-K{K}")
        lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + warning + r"Computational performance under the Bernoulli design for \(K=" + str(K) + r"\). Runtimes are mean seconds per generated matrix, including every preprocessing outcome.}", r"\label{" + label + "}", r"\end{table}"]
        tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"


def analyze_study(study_dir: Path, *, allow_partial=False, output_dir: Path | None = None) -> dict:
    root = Path(study_dir).resolve()
    manifest, data, issues = read_study(root, allow_partial=allow_partial)
    cells = summarize_cells(manifest, data)
    profile = manifest.get("study_profile", "paper")
    complete = not issues and all(r["n"] == r["expected_n"] for r in cells)
    full_status = "paper_complete" if profile == "paper" else "extension_complete"
    status = "partial" if not complete else (full_status if manifest["expected_replicates_per_cell"] == 1000 else "smoke_complete")
    suffix = "" if status in ("paper_complete", "extension_complete") else ("_PARTIAL" if status == "partial" else "_SMOKE")
    out = Path(output_dir) if output_dir else root / ("analysis" + suffix.lower())
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(study_id=manifest.get("study_id"), study_profile=profile, status=status,
                   paper_final=status == "paper_complete", study_complete=complete,
                   expected_replicates_per_cell=manifest["expected_replicates_per_cell"],
                   expected_total=sum(int(t["N"]) for t in manifest["tasks"]), actual_total=len(data),
                   source_tree_sha256=manifest["source_tree_sha256"],
                   manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                   runtime_provenance={k: v for k, v in manifest["_validated_runtime"].items() if k != "jobs"},
                   issues=issues, cells=cells,
                   computation_denominator="All generated matrices in the cell; not conditional on SAT invocation.",
                   conditional_denominator="Identifiable matrices in the cell; null when none is identifiable.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "runtime_environment.json").write_text(json.dumps(manifest["_validated_runtime"], indent=2, allow_nan=False) + "\n")
    csv_records = [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in r.items()} for r in cells]
    with (out / "cell_summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_records[0]))
        writer.writeheader(); writer.writerows(csv_records)
    table_name = "paper_tables" if profile == "paper" else "bernoulli_large_tables"
    (out / f"{table_name}{suffix}.tex").write_text(latex_tables(cells, status, profile))
    (out / "README.md").write_text(f"Study status: **{status}**.\n\nValidated {len(data)} of {summary['expected_total']} expected records.\n\n"
        "Percentages use the actual cell denominator; conditional percentages use the identifiable denominator. "
        "All runtime means include matrices resolved during preprocessing. Exact counts and unrounded values are in summary.json and cell_summary.csv.\n\n"
        "Software versions, timing definitions, CPU/hostname job counts, execution modes, allocated CPUs, affinity sizes, and thread settings are in runtime_environment.json. Different CPU models are explicitly flagged as mixed hardware.\n\n"
        + ("These are complete 1,000-replicate tables for the declared study profile.\n" if status in ("paper_complete", "extension_complete") else "These outputs are not final paper results.\n"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = analyze_study(args.study_dir, allow_partial=args.allow_partial, output_dir=args.output_dir)
    print(json.dumps({k: summary[k] for k in ("status", "paper_final", "actual_total", "expected_total")}))


if __name__ == "__main__":
    main()
