#!/bin/bash
#SBATCH --job-name=submit_runtime_expr
#SBATCH --output=logs/submit_runtime_%j.out
#SBATCH --time=48:00:00
#SBATCH --mem=4G
# submit_multiple.sh
# This script submits multiple jobs using run_runtime_expr.sh.
# Usage: ./submit_multiple.sh <J> <K> <nseeds>
# Example: ./submit_multiple.sh 5 4 100
# This will submit 100 jobs with N fixed to 10 and seeds from 0 to 99.

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <J> <K> <nseeds>"
    exit 1
fi

J=$1
K=$2
NSEEDS=$3
N=10  # fixed number of matrices per job

for (( seed=0; seed<NSEEDS; seed++ ))
do
    sbatch run_runtime_expr.sh "$J" "$K" "$N" "$seed"
done
