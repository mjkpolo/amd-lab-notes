#!/bin/bash
#SBATCH --job-name=sbatch_run_mb
#SBATCH --output=sbatch_run_mb_%j.out
##SBATCH --error=sbatch_run_mb_%j.err
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p mi3008x
#SBATCH -q alloc_diwu_05142024_06302025

set -x

cd $SLURM_SUBMIT_DIR
echo "RUNANDTIME_START $(date +%s)"
rm -rf FULL_MFMA STARTUP_MFMA
make -j
rocprofv3 --pmc SQ_VALU_MFMA_BUSY_CYCLES,GRBM_GUI_ACTIVE,SQ_INSTS_VALU_MFMA_MOPS_F16 -d FULL_MFMA -- ./mfma_fp32_16x16x16fp16
rocprofv3 --pmc SQ_VALU_MFMA_BUSY_CYCLES,GRBM_GUI_ACTIVE,SQ_INSTS_VALU_MFMA_MOPS_F16 -d STARTUP_MFMA -- ./mfma_fp32_16x16x16fp16_startup
./util.py
echo "RUNANDTIME_STOP $(date +%s)"
