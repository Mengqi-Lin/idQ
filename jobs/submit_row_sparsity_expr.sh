#!/usr/bin/env bash
# Run with bash on a login node; this submits one Slurm array.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash jobs/submit_row_sparsity_expr.sh [--dry-run] J K m nseeds [solver [min_row_size [row_size_distribution [row_size_probs]]]]

Defaults: IDQ_N=10, solver=glucose42, min_row_size=1,
row_size_distribution=uniform. Probabilities are comma-separated.
Example: bash jobs/submit_row_sparsity_expr.sh 50 10 3 100 glucose42 1 custom 0.4,0.4,0.2
EOF
}

idq_dry_run=false
args=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) idq_dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        *) args+=("$arg") ;;
    esac
done
set -- "${args[@]}"
if (( $# < 4 || $# > 8 )); then usage >&2; exit 2; fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
idq_setup_paths
J=$1 K=$2 m=$3 nseeds=$4 solver=${5:-glucose42}
min_row_size=${6:-1} row_size_distribution=${7:-uniform} row_size_probs=${8:-}
N=${IDQ_N:-10}
idq_positive_integer J "$J"
idq_positive_integer K "$K"
idq_positive_integer IDQ_N "$N"
idq_positive_integer m "$m"
idq_positive_integer min_row_size "$min_row_size"
worker_args=("$J" "$K" "$N" "$m" array "$solver" "$min_row_size" "$row_size_distribution")
[[ -z "$row_size_probs" ]] || worker_args+=("$row_size_probs")
idq_submit row_sparsity_expr.sh row_sparsity "$nseeds" "${worker_args[@]}"
