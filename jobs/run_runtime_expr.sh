#!/bin/bash
#SBATCH --job-name=runtime_expr
#SBATCH --output=logs/runtime_expr_%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=8G

# Usage: ./run_runtime_expr.sh <J> <K> <N> <seed>
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <J> <K> <N> <seed>"
    exit 1
fi

J=$1
K=$2
N=$3
SEED=$4

# Run the Python script with the provided parameters.
/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/runtime_expr.py "$J" "$K" "$N" "$SEED"


