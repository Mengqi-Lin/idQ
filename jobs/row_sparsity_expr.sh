#!/bin/bash
#SBATCH --job-name=row_sparsity_expr
#SBATCH --output=../logs/row_sparsity_expr_%j.out
#SBATCH --time=68:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7G
#SBATCH -p largemem,standard
# #SBATCH --licenses=gurobi@slurmdb:1


# Usage:
#   sbatch row_sparsity_expr.sh <J> <K> <N> <m> <seed> <solver> [min_row_size] [row_size_distribution] [row_size_probs]
#
# Examples:
#   sbatch row_sparsity_expr.sh 50 10 10 3 0 -1
#   sbatch row_sparsity_expr.sh 50 10 10 4 0 -1
#   sbatch row_sparsity_expr.sh 50 10 10 4 0 -1 2
#   sbatch row_sparsity_expr.sh 50 10 10 3 0 -1 1 custom 0.4,0.4,0.2

if [ "$#" -lt 6 ] || [ "$#" -gt 9 ]; then
    echo "Usage: $0 <J> <K> <N> <m> <seed> <solver> [min_row_size] [row_size_distribution] [row_size_probs]"
    exit 1
fi

module load gurobi/10.0.2

J=$1
K=$2
N=$3
m=$4
SEED=$5
solver=$6

# Optional arguments.
MIN_ROW_SIZE=${7:-1}
ROW_SIZE_DISTRIBUTION=${8:-uniform}
ROW_SIZE_PROBS=${9:-}

PYTHON=/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python
SCRIPT=/home/lemonkey/idQ/py/row_sparsity_expr.py

CMD=(
    "$PYTHON" "$SCRIPT"
    "$J" "$K" "$N" "$m" "$SEED" "$solver"
    --min-row-size "$MIN_ROW_SIZE"
    --row-size-distribution "$ROW_SIZE_DISTRIBUTION"
)

if [ -n "$ROW_SIZE_PROBS" ]; then
    CMD+=(--row-size-probs "$ROW_SIZE_PROBS")
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"