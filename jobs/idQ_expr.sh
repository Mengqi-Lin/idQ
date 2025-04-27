#!/bin/bash
#SBATCH --job-name=idQ_expr
#SBATCH --output=../logs/idQ_expr_%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=24G
#SBATCH --licenses=gurobi@slurmdb:1
#SBATCH --mem-per-cpu=2G  # with 8 CPUs → 16 GB total (bump to 3G if you need ~24 GB)


# Usage: ./run_runtime_expr.sh <J> <K> <N> <seed> <solver>
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <J> <K> <N> <p> <seed> <solver>"
    exit 1
fi

module load gurobi/10.0.2

J=$1
K=$2
N=$3
p=$4
SEED=$5
solver=$6
# Run the Python script with the provided parameters.
/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/idQ_expr.py "$J" "$K" "$N" "$p" "$SEED" "$solver"


