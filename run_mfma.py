#!/usr/bin/env python3

import subprocess
from subprocess import PIPE
from pathlib import Path
from os import chdir, getcwd

native = False


args = "-n 3 --num-compute-units=60 --cu-per-sa=15 --num-gpu-complex=4 --dgpu --gfx-version=gfx900 --reg-alloc-policy=dynamic --num-tccs=8 --bw-scalor=8 --num-dirs=64 --mem-size=16GB --vreg-file-size=16384 --sreg-file-size=800 --tcc-size=4MB --mem-type=HBM_2000_4H_1x64 --vrf_lm_bus_latency=63 --gpu-clock=1801MHz --max-coalesces-per-cycle=10 --max-cu-tokens=160 --mandatory_queue_latency=1 --mem-req-latency=69 --mem-resp-latency=69 --scalar-mem-req-latency=28 --sqc-size=16kB --TCC_latency=121 --tcc-assoc=16 --tcc-tag-access-latency=1 --tcc-data-access-latency=2 --atomic-alu-latency=58 --tcc-num-atomic-alus=256 --glc-atomic-latency=137 --memtime-latency=41"


def runcmd(*args):
    print(f"running {' '.join(args)}:", flush=True)
    pd = getcwd()
    chdir('..')
    stdout, stderr = subprocess.Popen(
        args, stdout=PIPE, stderr=PIPE).communicate()
    print(f"STDOUT:\n{stdout.decode('utf-8')}\n", flush=True)
    print(f"STDERR:\n{stderr.decode('utf-8')}\n", flush=True)
    chdir(pd)


def rundocker(*args):
    if native:
        runcmd("bash", "-c", " ".join(args))
    else:
        runcmd("docker", "run",
               "--rm",
               "--privileged",
               "-t",
               "-v", ".:/mnt",
               "-w", "/mnt",
               container_name,
               "bash", "-c", " ".join(args))


container_name = "mi200-container"
dockerfile_dir = "gem5/util/dockerfiles/gpu-fs"
bench_dir = Path("amd-lab-notes/matrix-cores")
resources = Path("gem5/gem5-resources/src/x86-ubuntu-gpu-ml")


def build_docker():
    runcmd("docker", "build", "-t", container_name, dockerfile_dir)


def compile():
    compile = f"make all -j -B -C {bench_dir}"
    rundocker(compile)


def main():
    if not native:
        build_docker()
    compile()
    bins = (tuple(file.stem for file in ('..' / bench_dir / 'src').iterdir()
                  if file.suffix == '.cpp'))
    for bin in bins:
        if native:
            runcmd(bench_dir/bin)
        else:
            rundocker("gem5/build/VEGA_X86/gem5.opt",
                      "gem5/configs/example/gpufs/mi200.py",
                      args,
                      "-a",
                      str(bench_dir/bin),
                      "--kernel",
                      str(resources/"vmlinux-gpu-ml"),
                      "--disk-image",
                      str(resources/"disk-image"/"x86-ubuntu-gpu-ml"))
            runcmd("cp", "m5out/system.pc.com_1.device",
                   f"{bin}_system.pc.com_1.device")
    return 0


if __name__ == "__main__":
    exit(main())
