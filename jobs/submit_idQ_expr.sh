#!/bin/bash
#SBATCH --job-name=submit_idQ_expr
#SBATCH --output=logs/submit_idQ_%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=16G
# submit_multiple.sh
# This script submits multiple jobs using idQ_expr.sh.
# This will submit 100 jobs with N fixed to 10 and seeds from 0 to 99.

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <J> <K> <p> <nseeds> <solver>"
    exit 1
fi

J=$1
K=$2
p=$3
NSEEDS=$4
solver=$5
N=1  # fixed number of simulations per job

for (( seed=0; seed<NSEEDS; seed++ ))
do
    sbatch idQ_expr.sh "$J" "$K" "$N" "$p" "$seed" "$solver"
done
