#!/bin/bash
#SBATCH --job-name=enumerate_global_idQ
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=8G

# Use the variables passed to the script to construct the output file name
OUTFILE="../log/enumerate_global_idQ_J${J}_K${K}_C${C}.out"

/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/simu_enumerate_global_idQ.py $J $K $C > $OUTFILE
