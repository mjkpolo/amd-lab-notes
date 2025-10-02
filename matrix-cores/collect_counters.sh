#!/usr/bin/env bash
#SBATCH --job-name=tiling_gemms
#SBATCH --output=sbatch_tiling_gemms_%j.out
##SBATCH --error=sbatch_tiling_gemms_%j.err
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 00:02:00
#SBATCH -p mi3008x
#SBATCH -q alloc_diwu_04012025_03312026


set -ex

if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

EXE="$@"
PROF_DIR="${PROF_DIR:-mfma_counters}"

if [[ -z "$EXE" ]]; then
  echo "Pass binary to execute" 1>&2
  exit 1
else
  echo "Passed binary: $EXE to profile" 1>&2
fi

ROCPROF_ARGS=(
  --pmc SQ_VALU_MFMA_BUSY_CYCLES,GRBM_GUI_ACTIVE
  -d "$PROF_DIR"
)

PROF_BIN=/opt/rocm-6.4.1/bin/rocprofv3

"$PROF_BIN" "${ROCPROF_ARGS[@]}" -- $EXE

