#!/usr/bin/env bash
#SBATCH --job-name=idQ_paper_analysis
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
study_dir=${1:?Supply an absolute study directory}
export IDQ_ROOT=${2:-${IDQ_ROOT:?Supply the idQ project directory as the second argument}}
source "$IDQ_ROOT/jobs/_common.sh"
idq_setup_paths
idq_set_python
export PYTHONPATH="$IDQ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$idq_python" -m idQ.experiments.paper_summary --study-dir "$study_dir"
