#!/bin/bash
#SBATCH --job-name=exprQ_J${1}_K${2}
#SBATCH --output="../log/exprQ_J${1}_K${2}.out"
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=8G

/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/expr_Q.py $1 $2 $3