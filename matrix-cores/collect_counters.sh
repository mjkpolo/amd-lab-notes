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

rm -rf ${PROF_DIR}_*

ROCPROF_ARGS=(
  "--pmc GRBM_GUI_ACTIVE,SQ_ACTIVE_INST_VALU,SQ_INSTS_LDS                                     -d ${PROF_DIR}_0"
  "--pmc SQ_INSTS_VALU_ADD_F16,SQ_INSTS_VALU_ADD_F32,SQ_INSTS_VALU_ADD_F64                    -d ${PROF_DIR}_1"
  "--pmc SQ_INSTS_VALU_FMA_F16,SQ_INSTS_VALU_FMA_F32,SQ_INSTS_VALU_FMA_F64                    -d ${PROF_DIR}_2"
  "--pmc SQ_INSTS_VALU_MFMA_MOPS_BF16,SQ_INSTS_VALU_MFMA_MOPS_F16,SQ_INSTS_VALU_MFMA_MOPS_F32 -d ${PROF_DIR}_3"
  "--pmc SQ_INSTS_VALU_MFMA_MOPS_F64,SQ_INSTS_VALU_MFMA_MOPS_I8,SQ_INSTS_VALU_MUL_F16         -d ${PROF_DIR}_4"
  "--pmc SQ_INSTS_VALU_MUL_F32,SQ_INSTS_VALU_MUL_F64,SQ_INSTS_VALU_TRANS_F16                  -d ${PROF_DIR}_5"
  "--pmc SQ_INSTS_VALU_TRANS_F32,SQ_INSTS_VALU_TRANS_F64,SQ_INSTS_VMEM                        -d ${PROF_DIR}_6"
  "--pmc SQ_LDS_BANK_CONFLICT,SQ_LDS_IDX_ACTIVE,SQ_VALU_MFMA_BUSY_CYCLES                      -d ${PROF_DIR}_7"
  "--pmc TCC_BUBBLE_sum,TCC_EA0_RDREQ_32B_sum,TCC_EA0_RDREQ_sum                               -d ${PROF_DIR}_8"
  "--pmc TCC_EA0_WRREQ_64B_sum,TCC_EA0_WRREQ_sum,TCC_HIT_sum                                  -d ${PROF_DIR}_9"
  "--pmc TCC_MISS_sum,TCC_REQ_sum,TCP_GATE_EN1_sum                                            -d ${PROF_DIR}_10"
  "--pmc TCP_GATE_EN2_sum,TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum,TCP_TCC_ATOMIC_WITH_RET_REQ_sum  -d ${PROF_DIR}_11"
  "--pmc TCP_TCC_READ_REQ_sum,TCP_TCC_WRITE_REQ_sum,TCP_TOTAL_CACHE_ACCESSES_sum              -d ${PROF_DIR}_12"
  "--pmc MeanOccupancyPerCU                                                                   -d ${PROF_DIR}_13"
  "--pmc SQC_TC_DATA_WRITE_REQ,TCP_TOTAL_ATOMIC_WITH_RET_sum,TCP_TOTAL_WRITE_sum              -d ${PROF_DIR}_14"
  "--pmc SQC_DCACHE_MISSES,SQC_ICACHE_HITS,SQ_ACTIVE_INST_VMEM                                -d ${PROF_DIR}_15"
  "--pmc SQC_DCACHE_HITS,TCC_EA0_ATOMIC_sum,SPI_CSN_WAVE                                      -d ${PROF_DIR}_16"
  "--pmc SPI_CSN_NUM_THREADGROUPS,TCC_EA0_WRREQ_DRAM_sum,TCP_TOTAL_READ_sum                   -d ${PROF_DIR}_17"
  "--pmc SQC_DCACHE_REQ,SQC_ICACHE_MISSES,SQC_DCACHE_MISSES_DUPLICATE                         -d ${PROF_DIR}_18"
  "--pmc SQ_BUSY_CU_CYCLES,SQC_TC_DATA_ATOMIC_REQ,TCC_BUSY_sum                                -d ${PROF_DIR}_19"
  "--pmc TCC_ATOMIC_sum,SQ_ACTIVE_INST_MISC,TCC_READ_sum                                      -d ${PROF_DIR}_20"
  "--pmc TCP_TOTAL_ATOMIC_WITHOUT_RET_sum,TCC_EA0_RDREQ_DRAM_sum,SQ_ACTIVE_INST_SCA           -d ${PROF_DIR}_21"
  "--pmc SQ_ACTIVE_INST_FLAT,SQC_ICACHE_REQ,SQC_TC_INST_REQ                                   -d ${PROF_DIR}_22"
  "--pmc TCC_WRITE_sum,SQC_ICACHE_MISSES_DUPLICATE,SQC_TC_DATA_READ_REQ                       -d ${PROF_DIR}_23"
)

PROF_BIN=/opt/rocm-6.4.1/bin/rocprofv3

for args in "${ROCPROF_ARGS[@]}"; do
  "$PROF_BIN" $args -- $EXE
done

