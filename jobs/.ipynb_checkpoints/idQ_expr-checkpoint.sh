#!/bin/bash
#SBATCH --job-name=idQ_expr
#SBATCH --output=logs/idQ_expr_%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=16G

# Usage: ./run_runtime_expr.sh <J> <K> <N> <seed>
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <J> <K> <N> <p> <seed>"
    exit 1
fi

J=$1
K=$2
N=$3
p=$4
SEED=$5

# Run the Python script with the provided parameters.
/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/idQ_expr.py "$J" "$K" "$N" "$p" "$SEED"


