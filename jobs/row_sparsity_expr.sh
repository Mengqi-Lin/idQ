#!/usr/bin/env bash
#SBATCH --job-name=idQ_row_sparsity
#SBATCH --time=68:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=56G
#SBATCH --partition=largemem,standard

set -euo pipefail
usage() {
    printf 'Usage: bash jobs/row_sparsity_expr.sh J K N m seed [solver [min_row_size [row_size_distribution [row_size_probs]]]]\n'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# < 5 || $# > 9 )); then usage >&2; exit 2; fi

if [[ -n "${IDQ_ROOT:-}" ]]; then
    source "$IDQ_ROOT/jobs/_common.sh"
else
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    source "$script_dir/_common.sh"
fi
idq_setup_paths
idq_set_python
J=$1 K=$2 N=$3 m=$4 solver=${6:-glucose42}
min_row_size=${7:-1} row_size_distribution=${8:-uniform} row_size_probs=${9:-}
idq_positive_integer J "$J"
idq_positive_integer K "$K"
idq_positive_integer N "$N"
idq_positive_integer m "$m"
idq_positive_integer min_row_size "$min_row_size"
idq_resolve_seed "$5"
idq_prepare_output row_sparsity
cd -- "$IDQ_ROOT"
command=("$idq_python" -m idQ.experiments.row_sparsity
    "$J" "$K" "$N" "$m" "$idq_seed" "$solver"
    --min-row-size "$min_row_size"
    --row-size-distribution "$row_size_distribution")
[[ -z "$row_size_probs" ]] || command+=(--row-size-probs "$row_size_probs")
command+=(--output-csv "$idq_output_csv")
idq_print_command "${command[@]}"
exec "${command[@]}"
