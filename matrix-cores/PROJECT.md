# Tinygrad

## Collect counters

```
PROF_DIR=tinygrad_counters DEBUG=3 AM=1 HIP=1 HALF=1 N=4096 BEAM=3 sbatch collect_counters.sh python ../tinygrad/extra/gemm/simple_matmul.py
```

## Get MFMA util

```
./mfma_util.py tinygrad_counters
```

# Tiled

## Tiled v1

- One CU does all the work

```
MFMA Util (%):
[0.0049347]
```

## Tiled v2

```
MFMA Util (%):
[2.39275386]
```

- Lift the row and col iteration into a grid

```
make
PROF_DIR=tiled_counters sbatch collect_counters.sh ./tiled_gemm
```

## Get MFMA util

```
./mfma_util.py tiled_counters
```

