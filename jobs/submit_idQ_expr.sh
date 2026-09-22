#!/usr/bin/env bash
# Run with bash on a login node; this submits one Slurm array.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash jobs/submit_idQ_expr.sh [--dry-run] J K p nseeds [solver]

Each seed runs IDQ_N simulations (default 10). Solver defaults to glucose42.
Set IDQ_MAX_CONCURRENT to limit the number of running array tasks.
Example: IDQ_MAX_CONCURRENT=20 bash jobs/submit_idQ_expr.sh 50 10 0.3 100
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
if (( $# < 4 || $# > 5 )); then usage >&2; exit 2; fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
idq_setup_paths
J=$1 K=$2 p=$3 nseeds=$4 solver=${5:-glucose42}
N=${IDQ_N:-10}
idq_positive_integer J "$J"
idq_positive_integer K "$K"
idq_positive_integer IDQ_N "$N"
idq_submit idQ_expr.sh bernoulli "$nseeds" "$J" "$K" "$N" "$p" array "$solver"
