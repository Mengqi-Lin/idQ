#!/usr/bin/env bash
# Run with bash on the login node; this helper submits one array for all settings.
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash jobs/submit_paper_simulation.sh [--dry-run] [--missing] STUDY_DIR

Defaults: 36 settings, 1000 matrices per setting, 100 matrices per task,
          360 array tasks, at most 20 running tasks, one CPU, 4G, 4 hours.
Settings: IDQ_REPLICATES, IDQ_PER_TASK, IDQ_BASE_SEED, IDQ_MAX_CONCURRENT,
          IDQ_PROFILE (paper, or bernoulli_large for K=20,30 only),
          IDQ_ACCOUNT (optional), IDQ_PARTITION, IDQ_MEM, IDQ_TIME,
          IDQ_CONSTRAINT (optional), IDQ_PYTHON, IDQ_AUTO_ANALYZE (default 1).
An analysis job is queued after successful completion of the array.
Every submission prints the manifest's profile, K values and replicate counts.
Explicit IDQ_PROFILE/REPLICATES/PER_TASK/BASE_SEED values must match an existing
manifest. Use a new study directory to change these settings.
--missing submits only missing/incomplete tasks. Ensure old jobs have stopped;
partial files are then archived before those tasks are rerun with the same seed.
EOF
}
dry_run=false
missing=false
positional=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --missing) missing=true ;;
        -h|--help) usage; exit 0 ;;
        *) positional+=("$arg") ;;
    esac
done
(( ${#positional[@]} == 1 )) || { usage >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export IDQ_ROOT=$(cd -- "$script_dir/.." && pwd -P)
source "$script_dir/_common.sh"
idq_setup_paths
idq_set_python
export PYTHONPATH="$IDQ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
study_dir=$("$idq_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${positional[0]}")
if [[ ! -f "$study_dir/manifest.json" ]]; then
    "$idq_python" -m idQ.experiments.paper_study prepare --study-dir "$study_dir" \
        --replicates "${IDQ_REPLICATES:-1000}" --per-task "${IDQ_PER_TASK:-100}" \
        --base-seed "${IDQ_BASE_SEED:-20260922}" --profile "${IDQ_PROFILE:-paper}"
fi
"$idq_python" - "$study_dir" <<'PY'
import os
import sys
from idQ.experiments.paper_study import load_manifest
from idQ.experiments.paper_summary import validate_manifest

root, manifest = load_manifest(sys.argv[1])
tasks = validate_manifest(manifest, root)
profile = manifest.get("study_profile", "paper")
requested_profile = os.environ.get("IDQ_PROFILE")
if requested_profile is not None and requested_profile != profile:
    sys.exit(f"Error: IDQ_PROFILE={requested_profile!r}, but {root / 'manifest.json'} "
             f"contains profile={profile!r}. Use a new study directory; no jobs submitted.")
for env_name, field in (("IDQ_REPLICATES", "expected_replicates_per_cell"),
                        ("IDQ_PER_TASK", "replicates_per_task"),
                        ("IDQ_BASE_SEED", "base_seed")):
    if env_name in os.environ:
        try:
            requested = int(os.environ[env_name])
        except ValueError:
            sys.exit(f"Error: {env_name} must be an integer; no jobs submitted.")
        if requested != manifest[field]:
            sys.exit(f"Error: {env_name}={requested}, but the existing manifest has "
                     f"{field}={manifest[field]}. Use a new study directory; no jobs submitted.")
print(f"Study: {root}")
print(f"Profile: {profile}; K values: {sorted({task['K'] for task in tasks})}")
print(f"Settings: {len({task['cell_id'] for task in tasks})}; "
      f"replicates per setting: {manifest['expected_replicates_per_cell']}; "
      f"matrices: {sum(task['N'] for task in tasks)}; tasks: {len(tasks)}")
PY
index_args=(--study-dir "$study_dir")
if [[ "$missing" == true ]]; then index_args+=(--missing); fi
indices=$("$idq_python" -m idQ.experiments.paper_study task-indices "${index_args[@]}")
if [[ -z "$indices" ]]; then printf 'No tasks require submission.\n'; exit 0; fi
concurrency=${IDQ_MAX_CONCURRENT:-20}
idq_positive_integer IDQ_MAX_CONCURRENT "$concurrency"
mkdir -p -- "$study_dir/logs"
export IDQ_PYTHON=$idq_python
export IDQ_RETRY_INCOMPLETE=0
if [[ "$missing" == true ]]; then export IDQ_RETRY_INCOMPLETE=1; fi
command=(sbatch --parsable --chdir "$IDQ_ROOT" --export ALL
         --array "$indices%$concurrency" --partition "${IDQ_PARTITION:-standard}"
         --mem "${IDQ_MEM:-4G}" --time "${IDQ_TIME:-04:00:00}"
         --output "$study_dir/logs/%A_%a.out" --error "$study_dir/logs/%A_%a.err")
if [[ -n "${IDQ_ACCOUNT:-}" ]]; then command+=(--account "$IDQ_ACCOUNT"); fi
if [[ -n "${IDQ_CONSTRAINT:-}" ]]; then command+=(--constraint "$IDQ_CONSTRAINT"); fi
command+=("$script_dir/paper_simulation.sh" "$study_dir")
idq_print_command "${command[@]}"
analysis_command=(sbatch --parsable --chdir "$IDQ_ROOT" --export ALL
                  --kill-on-invalid-dep=yes
                  --partition "${IDQ_PARTITION:-standard}"
                  --output "$study_dir/logs/analysis_%j.out"
                  --error "$study_dir/logs/analysis_%j.err")
if [[ -n "${IDQ_ACCOUNT:-}" ]]; then analysis_command+=(--account "$IDQ_ACCOUNT"); fi
if [[ "$dry_run" == false ]]; then
    submission=$("${command[@]}")
    printf 'Submitted array %s\n' "$submission"
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$submission" >> "$study_dir/submissions.tsv"
    if [[ "${IDQ_AUTO_ANALYZE:-1}" == 1 ]]; then
        analysis_command+=(--dependency "afterok:${submission%%;*}"
                          "$script_dir/analyze_paper_simulation.sh" "$study_dir" "$IDQ_ROOT")
        analysis_submission=$("${analysis_command[@]}")
        printf 'Submitted analysis job %s (after array success)\n' "$analysis_submission"
        printf '%s\tanalysis:%s\n' "$(date -u +%FT%TZ)" "$analysis_submission" >> "$study_dir/submissions.tsv"
    fi
elif [[ "${IDQ_AUTO_ANALYZE:-1}" == 1 ]]; then
    analysis_command+=(--dependency 'afterok:ARRAY_JOB_ID'
                      "$script_dir/analyze_paper_simulation.sh" "$study_dir" "$IDQ_ROOT")
    idq_print_command "${analysis_command[@]}"
fi
