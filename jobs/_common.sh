#!/usr/bin/env bash
# Shared helpers. This file is sourced by the four public job scripts.

idq_error() { printf 'Error: %s\n' "$*" >&2; exit 2; }

idq_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || idq_error "$1 must be a positive integer (got '$2')."
}

idq_setup_paths() {
    local script_dir
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    IDQ_ROOT=${IDQ_ROOT:-$(cd -- "$script_dir/.." && pwd -P)}
    [[ -d "$IDQ_ROOT/src/idQ" ]] || idq_error "IDQ_ROOT does not point to the idQ project: $IDQ_ROOT"
    IDQ_ROOT=$(cd -- "$IDQ_ROOT" && pwd -P)
    IDQ_DATA_DIR=${IDQ_DATA_DIR:-$IDQ_ROOT/data}
    [[ "$IDQ_DATA_DIR" = /* ]] || IDQ_DATA_DIR="$IDQ_ROOT/$IDQ_DATA_DIR"
    export IDQ_ROOT IDQ_DATA_DIR
    mkdir -p -- "$IDQ_ROOT/logs" "$IDQ_DATA_DIR"
}

idq_set_python() {
    if [[ -n "${IDQ_PYTHON:-}" ]]; then
        idq_python=$IDQ_PYTHON
    elif [[ -x "$IDQ_ROOT/.venv/bin/python" ]]; then
        idq_python="$IDQ_ROOT/.venv/bin/python"
    else
        idq_python=python3
    fi
    command -v -- "$idq_python" >/dev/null 2>&1 || idq_error "Python executable not found: $idq_python"
    # Each experiment is serial; avoid oversubscribing the allocated CPU.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
}

idq_resolve_seed() {
    idq_seed=$1
    if [[ "$idq_seed" == array ]]; then
        idq_seed=${SLURM_ARRAY_TASK_ID:?The 'array' seed requires a Slurm array task.}
    fi
    [[ "$idq_seed" =~ ^[0-9]+$ ]] || idq_error "seed must be a nonnegative integer or 'array'."
}

idq_prepare_output() {
    local experiment=$1 run_id prefix
    if [[ "$experiment" == row_sparsity ]]; then prefix=rowsparse; else prefix=bernoulli; fi
    run_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$(date -u +%Y%m%dT%H%M%SZ)_$$}}
    idq_output_dir="$IDQ_DATA_DIR/raw/$experiment/run_$run_id"
    mkdir -p -- "$idq_output_dir"
    idq_output_csv="$idq_output_dir/${prefix}_seed_${idq_seed}_diag.csv"
}

idq_print_command() { printf '%q ' "$@"; printf '\n'; }

idq_submit() {
    local worker=$1 experiment=$2 nseeds=$3
    shift 3
    idq_positive_integer nseeds "$nseeds"
    local array="0-$((nseeds - 1))"
    if [[ -n "${IDQ_MAX_CONCURRENT:-}" ]]; then
        idq_positive_integer IDQ_MAX_CONCURRENT "$IDQ_MAX_CONCURRENT"
        array+="%$IDQ_MAX_CONCURRENT"
    fi
    # Slurm opens log files before the worker starts, so create these here.
    mkdir -p -- "$IDQ_ROOT/logs" "$IDQ_DATA_DIR/raw/$experiment"
    local -a command=(
        env "IDQ_ROOT=$IDQ_ROOT" "IDQ_DATA_DIR=$IDQ_DATA_DIR"
        sbatch --chdir "$IDQ_ROOT" --export ALL
        --array "$array"
        --output "$IDQ_ROOT/logs/${experiment}_%A_%a.out"
        --error "$IDQ_ROOT/logs/${experiment}_%A_%a.err"
        "$IDQ_ROOT/jobs/$worker" "$@"
    )
    idq_print_command "${command[@]}"
    if [[ "$idq_dry_run" == false ]]; then
        "${command[@]}"
    fi
}
