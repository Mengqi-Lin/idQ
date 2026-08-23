#!/bin/bash
#SBATCH --job-name=submit_row_sparsity_expr
#SBATCH --output=logs/submit_row_sparsity_%j.out
#SBATCH --time=68:00:00
#SBATCH --mem=7G


# Usage:
#   sbatch submit_row_sparsity_expr.sh <J> <K> <m> <nseeds> <solver> [min_row_size] [row_size_distribution] [row_size_probs]
#
# Examples:
#   sbatch submit_row_sparsity_expr.sh 50 10 3 100 1
#   sbatch submit_row_sparsity_expr.sh 50 10 4 100 1
#
# Exclude pure nodes by construction:
#   sbatch submit_row_sparsity_expr.sh 50 10 4 100 1 2
#
# Custom D_j distribution over {1,2,3}:
#   sbatch submit_row_sparsity_expr.sh 50 10 3 100 1 1 custom 0.4,0.4,0.2

if [ "$#" -lt 5 ] || [ "$#" -gt 8 ]; then
    echo "Usage: $0 <J> <K> <m> <nseeds> <solver> [min_row_size] [row_size_distribution] [row_size_probs]"
    exit 1
fi

J=$1
K=$2
m=$3
NSEEDS=$4
solver=$5

# Optional arguments.
MIN_ROW_SIZE=${6:-1}
ROW_SIZE_DISTRIBUTION=${7:-uniform}
ROW_SIZE_PROBS=${8:-}

N=10  # fixed number of simulations per seed/job

for (( seed=0; seed<NSEEDS; seed++ ))
do
    if [ -n "$ROW_SIZE_PROBS" ]; then
        sbatch row_sparsity_expr.sh \
            "$J" "$K" "$N" "$m" "$seed" "$solver" \
            "$MIN_ROW_SIZE" "$ROW_SIZE_DISTRIBUTION" "$ROW_SIZE_PROBS"
    else
        sbatch row_sparsity_expr.sh \
            "$J" "$K" "$N" "$m" "$seed" "$solver" \
            "$MIN_ROW_SIZE" "$ROW_SIZE_DISTRIBUTION"
    fi
done