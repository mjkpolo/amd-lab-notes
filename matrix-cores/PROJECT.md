# Tinygrad

## Collect counters

```
PROF_DIR=tinygrad_counters DEBUG=3 AM=1 HIP=1 HALF=1 N=4096 BEAM=3 sbatch profile_flops.sh python ../tinygrad/extra/gemm/simple_matmul.py
```

## Get MFMA util

```
./mfma_util.py tinygrad_counters
```

# Tiled v1

```
make
PROF_DIR=tiled_v1_counters sbatch profile_flops.sh ./tiled_gemm_v1
```

## Get MFMA util

```
./mfma_util.py tiled_v1_counters
```

