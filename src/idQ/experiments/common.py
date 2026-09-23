"""Shared, reproducible experiment utilities for the conjunctive algorithm."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..core import (
    BRANCH_LABELS,
    contains_identity_submatrix,
    has_any_pure_node,
    identify,
    violates_three_column_necessary,
    violates_two_column_necessary,
)
from ..sat import normalize_cardinality_encoding, normalize_solver_name
from ..utils import representative_supports, validate_binary_matrix


SCHEMA_VERSION = 4


TIMING_DEFINITIONS = {
    "algorithm_time": (
        "Wall-clock time inside identify(), starting after input/option validation; "
        "includes basis reduction, executed submatrix checks, and the SAT path "
        "(CNF construction, solver setup/solve, certificate checking and lifting). "
        "Excludes matrix generation, original-matrix diagnostics, response-class "
        "counting, provenance, and CSV I/O."
    ),
    "basis_time": "Wall-clock time for basis reduction within identify().",
    "identity_check_time": "Wall-clock time for the executed identity-submatrix check.",
    "two_col_check_time": "Wall-clock time for the executed two-column check; zero if skipped.",
    "three_col_check_time": "Wall-clock time for the executed three-column check; zero if skipped.",
    "sat_time": (
        "Wall-clock SAT-path time reported by identify(), including CNF construction, "
        "solver setup/solve and SAT certificate checks; not solver-only time. "
        "Zero when SAT is skipped."
    ),
    "preprocess_time": "Sum of basis_time and the three recorded submatrix-check times.",
    "generation_time": "Wall-clock time for sampling and validating the original Q; outside algorithm_time.",
    "diagnostics_time": (
        "Wall-clock time after identify() for original-Q flags, M_basis, and other "
        "diagnostics; outside algorithm_time."
    ),
}


def source_provenance() -> dict[str, Any]:
    """Hash the imported package sources with one recipe shared by manifests."""
    package_root = Path(__file__).resolve().parents[1]
    hashes = {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"source_hashes": hashes, "source_tree_sha256": hashlib.sha256(payload).hexdigest()}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_environment() -> dict[str, Any]:
    """Collect local, read-only provenance without requiring platform utilities."""
    import idQ

    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        cpu_model = next(
            (line.split(":", 1)[1].strip() for line in cpu_info.splitlines()
             if line.startswith("model name") and ":" in line), None,
        )
    except OSError:
        cpu_model = None
    try:
        completed = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=3, check=False,
        )
        lscpu = completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        lscpu = None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    slurm_names = (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "SLURM_MEM_PER_NODE",
    )
    return {
        "python_version": platform.python_version(), "python_full_version": sys.version,
        "python_executable": sys.executable,
        "pysat_version": _package_version("python-sat"),
        "numpy_version": np.__version__, "idQ_version": idQ.__version__,
        "idQ_installed_distribution_version": _package_version("idQ"),
        "idQ_source_path": str(Path(idQ.__file__).resolve()),
        "platform": platform.platform(), "processor": platform.processor(),
        "machine": platform.machine(), "hostname": socket.gethostname(),
        "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity, "lscpu": lscpu,
        "slurm": {name: os.environ[name] for name in slurm_names if name in os.environ},
        "thread_environment": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if name in os.environ
        },
    }


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_dimensions(J: int, K: int, N: int | None = None) -> None:
    if not isinstance(J, (int, np.integer)) or int(J) <= 0:
        raise ValueError("J must be a positive integer.")
    if not isinstance(K, (int, np.integer)) or int(K) <= 0:
        raise ValueError("K must be a positive integer.")
    if N is not None and (not isinstance(N, (int, np.integer)) or int(N) <= 0):
        raise ValueError("N must be a positive integer.")


def make_rng(seed: int, engine: str = "legacy"):
    """Create a local RNG.

    ``legacy`` uses the MT19937 ``RandomState`` stream from the original
    ``idQ_expr.py`` and therefore preserves published-run reproducibility.
    ``pcg64`` selects NumPy's modern ``Generator`` stream.
    """
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    normalized = str(engine).strip().lower()
    if normalized == "legacy":
        return np.random.RandomState(int(seed))
    if normalized == "pcg64":
        return np.random.default_rng(int(seed))
    raise ValueError("rng_engine must be either 'legacy' or 'pcg64'.")


def rng_label(engine: str) -> str:
    normalized = str(engine).strip().lower()
    if normalized == "legacy":
        return "MT19937-RandomState"
    if normalized == "pcg64":
        return "PCG64-Generator"
    raise ValueError("rng_engine must be either 'legacy' or 'pcg64'.")


def sample_bernoulli_q(
    J: int,
    K: int,
    p: float,
    rng,
    *,
    condition_nonzero_rows: bool = True,
) -> np.ndarray:
    """Sample Bernoulli rows, optionally conditional on every row being nonzero.

    Conditioning is the generic default. Set the flag to false for unrestricted
    iid Bernoulli entries, as specified by the paper's Bernoulli experiment.
    """
    _validate_dimensions(J, K)
    if not np.isfinite(p) or not (0 <= float(p) <= 1):
        raise ValueError("p must be a finite number in [0, 1].")
    if condition_nonzero_rows and p == 0:
        raise ValueError("p=0 cannot generate nonzero rows.")

    Q = np.asarray(rng.binomial(1, float(p), size=(J, K)), dtype=int)
    if condition_nonzero_rows:
        zero_rows = np.flatnonzero(Q.sum(axis=1) == 0)
        while zero_rows.size:
            Q[zero_rows] = rng.binomial(1, float(p), size=(zero_rows.size, K))
            zero_rows = zero_rows[Q[zero_rows].sum(axis=1) == 0]
    return Q


def _normalized_probabilities(
    probabilities: Sequence[float],
    expected_length: int,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or len(probs) != expected_length:
        raise ValueError(f"Expected {expected_length} row-size probabilities.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("row_size_probs must be finite and nonnegative.")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("row_size_probs must have positive sum.")
    return probs / total


def sample_row_sparse_q(
    J: int,
    K: int,
    m: int,
    rng,
    *,
    min_row_size: int = 1,
    row_size_distribution: str = "uniform",
    row_size_probs: Sequence[float] | None = None,
) -> np.ndarray:
    """Sample a row-sparse Q matrix with explicit row-size distribution."""
    _validate_dimensions(J, K)
    if not isinstance(m, (int, np.integer)) or int(m) < 0:
        raise ValueError("m must be a nonnegative integer.")
    if not isinstance(min_row_size, (int, np.integer)) or int(min_row_size) < 0:
        raise ValueError("min_row_size must be a nonnegative integer.")

    m_eff = min(int(m), int(K))
    min_eff = int(min_row_size)
    if min_eff > m_eff:
        raise ValueError("min_row_size must not exceed min(m, K).")

    distribution = str(row_size_distribution).strip().lower()
    full_support = np.arange(min_eff, m_eff + 1, dtype=int)
    if distribution == "uniform":
        if row_size_probs is not None:
            raise ValueError("row_size_probs is only valid for distribution='custom'.")
        sizes = rng.choice(full_support, size=J, replace=True)
    elif distribution == "fixed":
        if row_size_probs is not None:
            raise ValueError("row_size_probs is only valid for distribution='custom'.")
        sizes = np.full(J, m_eff, dtype=int)
    elif distribution == "custom":
        if row_size_probs is None:
            raise ValueError("row_size_probs is required for distribution='custom'.")
        probs = _normalized_probabilities(row_size_probs, len(full_support))
        sizes = rng.choice(full_support, size=J, replace=True, p=probs)
    else:
        raise ValueError("row_size_distribution must be uniform, fixed, or custom.")

    Q = np.zeros((J, K), dtype=int)
    for j, size in enumerate(sizes):
        if int(size) > 0:
            attributes = rng.choice(K, size=int(size), replace=False)
            Q[j, attributes] = 1
    return Q


def actual_row_size_metadata(
    *,
    K: int,
    m: int,
    min_row_size: int,
    row_size_distribution: str,
    row_size_probs: Sequence[float] | None,
) -> dict[str, Any]:
    """Return metadata describing the distribution actually sampled."""
    if not isinstance(K, (int, np.integer)) or int(K) <= 0:
        raise ValueError("K must be a positive integer.")
    if not isinstance(m, (int, np.integer)) or int(m) < 0:
        raise ValueError("m must be a nonnegative integer.")
    if (
        not isinstance(min_row_size, (int, np.integer))
        or int(min_row_size) < 0
    ):
        raise ValueError("min_row_size must be a nonnegative integer.")
    m_eff = min(int(m), int(K))
    if int(min_row_size) > m_eff:
        raise ValueError("min_row_size must not exceed min(m, K).")
    distribution = str(row_size_distribution).strip().lower()
    full_support = list(range(int(min_row_size), m_eff + 1))
    if distribution == "fixed":
        if row_size_probs is not None:
            raise ValueError("row_size_probs is only valid for distribution='custom'.")
        support = [m_eff]
        probabilities = [1.0]
    elif distribution == "uniform":
        if row_size_probs is not None:
            raise ValueError("row_size_probs is only valid for distribution='custom'.")
        support = full_support
        probabilities = [1.0 / len(support)] * len(support)
    elif distribution == "custom":
        if row_size_probs is None:
            raise ValueError("row_size_probs is required for distribution='custom'.")
        support = full_support
        probabilities = _normalized_probabilities(
            row_size_probs,
            len(support),
        ).tolist()
    else:
        raise ValueError("row_size_distribution must be uniform, fixed, or custom.")
    return {
        "m_requested": int(m),
        "m_effective": int(m_eff),
        "min_row_size_requested": int(min_row_size),
        "row_size_distribution": distribution,
        "row_size_support": support,
        "row_size_probs": probabilities,
    }


def parse_probability_list(prob_string: str | None) -> list[float] | None:
    """Parse comma-separated probabilities from a CLI option."""
    if prob_string is None or not str(prob_string).strip():
        return None
    try:
        values = [float(value.strip()) for value in str(prob_string).split(",")]
    except ValueError as exc:
        raise ValueError("row-size probabilities must be comma-separated numbers.") from exc
    if not values:
        raise ValueError("row-size probabilities must not be empty.")
    return values


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    return value


def count_representative_classes(Q_basis: np.ndarray) -> int:
    """Compute ``|R(Q_basis)|`` through OR closure, not a ``2**K`` tensor."""
    basis = np.asarray(Q_basis)
    if basis.ndim != 2:
        raise ValueError("Q_basis must be two-dimensional.")
    if basis.shape[0] == 0:
        return 1
    return len(representative_supports(basis))


def identifiability_expr(
    Q: np.ndarray,
    solver: str | int = "glucose42",
    *,
    verbose: bool = False,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
    compute_class_count: bool = True,
) -> tuple[int, np.ndarray | None, int, dict[str, Any]]:
    """Run the one canonical core algorithm and add experiment diagnostics."""
    array = validate_binary_matrix(Q)
    requested_encoding = normalize_cardinality_encoding(cardinality_encoding)
    result = identify(
        array, solver_name=solver, verbose=verbose,
        cardinality_encoding=requested_encoding, maximal_candidate=maximal_candidate,
    )
    diagnostics_start = time.perf_counter()
    basis = result.basis
    assert basis is not None

    basis_nonempty = basis.shape[0] > 0
    timings = result.timings
    identity_original = contains_identity_submatrix(array)
    pure_original = has_any_pure_node(array)
    two_violation = violates_two_column_necessary(array)
    three_violation = violates_three_column_necessary(array)
    both_pass = int(not two_violation and not three_violation)
    diagnostics: dict[str, Any] = {
        **timings,
        "preprocess_time": sum(
            timings[name]
            for name in (
                "basis_time",
                "identity_check_time",
                "two_col_check_time",
                "three_col_check_time",
            )
        ),
        "J_basis": int(basis.shape[0]),
        "M_basis": count_representative_classes(basis) if compute_class_count else None,
        "M_basis_computed": int(bool(compute_class_count)),
        "identifiable": result.status,
        "branch": int(result.branch),
        "branch_label": result.branch_label,
        "sat_called": int(result.branch in (5, 6)),
        "sat_formulation": result.sat_formulation or "",
        "sat_cardinality_encoding": result.sat_cardinality_encoding or "",
        "cardinality_encoding_requested": requested_encoding,
        "maximal_candidate_requested": int(bool(maximal_candidate)),
        "sat_variables": int(result.sat_variables),
        "sat_clauses": int(result.sat_clauses),
        "solver_name": normalize_solver_name(solver),
        "identity_original": int(identity_original),
        "not_complete_original": int(not identity_original),
        "has_pure_node_original": int(pure_original),
        "no_pure_nodes_original": int(not pure_original),
        "has_zero_row_original": int(np.any(array.sum(axis=1) == 0)),
        "two_col_violation_original": int(two_violation),
        "three_col_violation_original": int(three_violation),
        "two_column_pass_original": int(not two_violation),
        "both_necessary_checks_pass_original": both_pass,
        "I_3c": both_pass,
        "identity_basis": int(
            basis_nonempty and contains_identity_submatrix(basis)
        ),
        "two_col_violation_basis": int(
            (basis.shape[1] >= 2 and not basis_nonempty)
            or (basis_nonempty and violates_two_column_necessary(basis))
        ),
    }
    diagnostics["diagnostics_time"] = time.perf_counter() - diagnostics_start
    return result.status, result.counterexample, result.branch, diagnostics


def run_design_expr(
    *,
    J: int,
    K: int,
    N: int,
    seed: int,
    solver: str | int,
    sampler: Callable[[Any], np.ndarray],
    metadata: Mapping[str, Any],
    output_csv: str | os.PathLike[str],
    rng_engine: str = "legacy",
    overwrite: bool = False,
    verbose: bool = False,
    checkpoint_every: int = 10,
    cardinality_encoding: str = "exclude_x",
    maximal_candidate: bool = False,
    compute_class_count: bool = True,
) -> list[dict[str, Any]]:
    """Run one seed/job and atomically publish its CSV after successful completion."""
    _validate_dimensions(J, K, N)
    if not isinstance(checkpoint_every, (int, np.integer)) or checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer.")
    normalized_solver = normalize_solver_name(solver)
    requested_encoding = normalize_cardinality_encoding(cardinality_encoding)
    rng = make_rng(seed, rng_engine)
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass overwrite=True explicitly."
        )

    temporary_path = output_path.with_name(output_path.name + ".part")
    metadata_path = output_path.with_name(output_path.name + ".metadata.json")
    if temporary_path.exists():
        raise FileExistsError(
            f"Incomplete output already exists: {temporary_path}. Inspect or remove it first."
        )
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Run metadata already exists: {metadata_path}. Inspect it before overwriting."
        )

    run_metadata = {
        **dict(metadata),
        "schema_version": SCHEMA_VERSION, "status": "running", "completed": False,
        "completed_replicates": 0,
        "J": int(J), "K": int(K), "N": int(N), "seed": int(seed),
        "operator": "conj", "solver_name": normalized_solver,
        "rng_engine": rng_label(rng_engine),
        "cardinality_encoding_requested": requested_encoding,
        "maximal_candidate_requested": bool(maximal_candidate),
        "compute_class_count": bool(compute_class_count),
        "checkpoint_every": int(checkpoint_every),
        "output_csv": str(output_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _runtime_environment(),
        "timing_definitions": TIMING_DEFINITIONS,
        "matrix_encoding": {
            "q_bitstring": "Original Q entries in row-major order, ASCII 0/1, length J*K.",
            "q_sha256": "SHA256 of ASCII str(J) + ',' + str(K) + ':' + q_bitstring.",
        },
        **source_provenance(),
    }

    rows: list[dict[str, Any]] = []
    try:
        with temporary_path.open("x", newline="", encoding="utf-8") as csvfile:
            _write_metadata(metadata_path, run_metadata)
            writer: csv.DictWriter | None = None
            for simulation_index in range(int(N)):
                generation_start = time.perf_counter()
                Q = validate_binary_matrix(sampler(rng))
                if Q.shape != (int(J), int(K)):
                    raise ValueError(
                        f"Sampler returned shape {Q.shape}; expected {(int(J), int(K))}."
                    )
                generation_time = time.perf_counter() - generation_start

                _, _, _, diagnostics = identifiability_expr(
                    Q,
                    solver=normalized_solver,
                    verbose=verbose,
                    cardinality_encoding=requested_encoding,
                    maximal_candidate=maximal_candidate,
                    compute_class_count=compute_class_count,
                )
                bitstring = "".join(str(int(bit)) for bit in Q.ravel())
                q_hash = hashlib.sha256(f"{int(J)},{int(K)}:{bitstring}".encode("ascii")).hexdigest()
                row: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "J": int(J),
                    "K": int(K),
                    "N": int(N),
                    "seed": int(seed),
                    "sim": simulation_index,
                    "operator": "conj",
                    "solver_name": normalized_solver,
                    "rng_engine": rng_label(rng_engine),
                    "generation_time": generation_time,
                    "q_bitstring": bitstring,
                    "q_sha256": q_hash,
                }
                row.update({key: _to_csv_value(value) for key, value in metadata.items()})
                row.update({key: _to_csv_value(value) for key, value in diagnostics.items()})

                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)

                if (simulation_index + 1) % int(checkpoint_every) == 0:
                    csvfile.flush()
                    os.fsync(csvfile.fileno())
                    run_metadata["completed_replicates"] = len(rows)
                    run_metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    _write_metadata(metadata_path, run_metadata)

            csvfile.flush()
            os.fsync(csvfile.fileno())
        run_metadata.update(
            status="completed", completed=True, completed_replicates=len(rows),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_metadata(metadata_path, run_metadata)
        os.replace(temporary_path, output_path)
    except BaseException as exc:
        # A .part file is deliberately retained as evidence of an interrupted
        # job; summary scripts ignore it and a rerun cannot overwrite it silently.
        run_metadata.update(
            status="interrupted", completed=False, completed_replicates=len(rows),
            interrupted_at_utc=datetime.now(timezone.utc).isoformat(),
            exception_type=type(exc).__name__,
        )
        try:
            _write_metadata(metadata_path, run_metadata)
        except OSError:
            pass  # Preserve the original exception if metadata storage also fails.
        raise

    return rows


__all__ = [
    "BRANCH_LABELS",
    "SCHEMA_VERSION",
    "TIMING_DEFINITIONS",
    "actual_row_size_metadata",
    "count_representative_classes",
    "identifiability_expr",
    "make_rng",
    "parse_probability_list",
    "run_design_expr",
    "sample_bernoulli_q",
    "sample_row_sparse_q",
    "source_provenance",
]
