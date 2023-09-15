#!/bin/bash
#SBATCH --job-name=fraction_subtraction_data_Q
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --out=../log/fraction_subtraction_data_Q.out
#SBATCH --error=../log/fraction_subtraction_data_Q.err
/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/fraction_subtraction_data_Q.py 
