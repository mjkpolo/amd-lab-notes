#!/usr/bin/env python3

import subprocess
from pathlib import Path

native = False


def runcmd(*args):
    subprocess.Popen(args).communicate()


def rundocker(*args):
    if native:
        runcmd("bash", "-c", " ".join(args))
    else:
        runcmd("docker", "run",
               "--rm",
               "--privileged",
               "-t",
               "-v", ".:/gem5",
               "-w", "/gem5",
               container_name,
               "bash", "-c", " ".join(args))


container_name = "mi200-container"
dockerfile_dir = "../gem5/util/dockerfiles/gpu-fs"
bench_dir = Path("matrix-cores")
resources = Path("../gem5-resources/src/x86-ubuntu-gpu-ml")


def build_docker():
    runcmd("docker", "build", "-t", container_name, dockerfile_dir)


def compile():
    compile = f"make all -j -B -C {bench_dir}"
    rundocker(compile)


def main():
    if not native:
        build_docker()
    compile()
    bins = (tuple(file.stem for file in (bench_dir / 'src').iterdir()
                  if file.suffix == '.cpp'))
    for bin in bins:
        print(f"running {bin}:")
        rundocker("../gem5/build/VEGA_X86/gem5.opt",
                  "configs/example/gpufs/mi200.py",
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
