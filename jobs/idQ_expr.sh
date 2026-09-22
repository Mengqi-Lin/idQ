#!/usr/bin/env bash
#SBATCH --job-name=idQ_bernoulli
#SBATCH --time=68:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --partition=largemem,standard

set -euo pipefail
usage() {
    printf 'Usage: bash jobs/idQ_expr.sh J K N p seed [solver]\n'
    printf "Use 'array' for seed only when launched by the submit helper.\n"
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# < 5 || $# > 6 )); then usage >&2; exit 2; fi

# Slurm copies this script to its spool directory. The submit helper exports
# IDQ_ROOT so this lookup does not depend on the copied script's location.
if [[ -n "${IDQ_ROOT:-}" ]]; then
    source "$IDQ_ROOT/jobs/_common.sh"
else
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    source "$script_dir/_common.sh"
fi
idq_setup_paths
idq_set_python
J=$1 K=$2 N=$3 p=$4 solver=${6:-glucose42}
idq_positive_integer J "$J"
idq_positive_integer K "$K"
idq_positive_integer N "$N"
idq_resolve_seed "$5"
idq_prepare_output bernoulli
cd -- "$IDQ_ROOT"
command=("$idq_python" -m idQ.experiments.bernoulli
    "$J" "$K" "$N" "$p" "$idq_seed" "$solver"
    --output-csv "$idq_output_csv")
idq_print_command "${command[@]}"
exec "${command[@]}"
