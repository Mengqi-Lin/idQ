#!/usr/bin/env bash
#SBATCH --job-name=idQ_paper
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail

: "${IDQ_ROOT:?Use jobs/submit_paper_simulation.sh to submit this worker.}"
source "$IDQ_ROOT/jobs/_common.sh"
idq_setup_paths
idq_set_python
export PYTHONPATH="$IDQ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
study_dir=${1:?Missing study directory}
task_id=${2:-${SLURM_ARRAY_TASK_ID:?Missing array task ID}}
args=("$idq_python" -m idQ.experiments.paper_study run-task
      --study-dir "$study_dir" --task-id "$task_id")
if [[ "${IDQ_RETRY_INCOMPLETE:-0}" == 1 ]]; then args+=(--retry-incomplete); fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "${args[@]}"
else
    exec "${args[@]}"
fi
