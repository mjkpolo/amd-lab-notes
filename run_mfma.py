#!/usr/bin/env python3

import subprocess
from pathlib import Path


def runcmd(*args):
    subprocess.Popen(args).communicate()


container_name = "mi200-container"
bench_dir = Path("matrix-cores")


def compile():
    compile = f"make all -j -B -C {bench_dir}"
    runcmd("bash", "-c", compile)


def main():
    compile()
    bins = (tuple(file.stem for file in (bench_dir / 'src').iterdir()
                  if file.suffix == '.cpp'))
    for bin in bins:
        print(f"running {bin}:")
        runcmd(bench_dir / bin)
    return 0


if __name__ == "__main__":
    exit(main())
