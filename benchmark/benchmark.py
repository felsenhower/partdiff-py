#!/usr/bin/env python3

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

GAUSS_SEIDEL_ARGS: tuple[str, ...] = ("1", "1", "100", "2", "2", "100")
JACOBI_ARGS: tuple[str, ...] = ("1", "2", "100", "2", "2", "100")

NUM_REPETITIONS = 3

OUTPUT_FILE = Path("benchmark_results.csv").resolve()

@dataclass(frozen=True)
class Variant:
    path: Path
    commandline: tuple[str, ...]
    do_warmup: bool


def format_row(
    variant: str,
    i: int,
    method: str,
    runtime_internal: float,
    runtime_total: float,
) -> str:
    return (
        f"{variant:<12}, {i}, {method:<11},"
        f" {runtime_internal:>16.4f}, {runtime_total:>13.4f}"
    )


def run_and_measure(
    cmd: tuple[str, ...],
    args: tuple[str, ...],
    cwd: Path,
) -> tuple[float, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        (*cmd, *args),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    end = time.perf_counter()
    runtime_total = end - start
    runtime_internal = float("nan")
    for line in proc.stdout.splitlines():
        if "Calculation time" in line:
            runtime_internal = float(line.split()[2])
            break
    return runtime_internal, runtime_total


def status_line(variant: str, method: str, i: int) -> None:
    print(
        f'\rRunning (variant="{variant}", method="{method}", i={i})...',
        end="",
        flush=True,
    )


def clear_status_line() -> None:
    print("\r" + " " * 80 + "\r", end="", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Error: Must supply path to reference implementation!")
        sys.exit(1)
    reference = Path(sys.argv[1]).resolve()
    variants: dict[str, Variant] = {
        "reference": Variant(
            path=reference.parent,
            commandline=(str(reference),),
            do_warmup=False,
        ),
        "simple": Variant(
            path=Path("../simple"),
            commandline=("uv", "run", "--python", "cpython3.13", "main.py"),
            do_warmup=False,
        ),
        "nuitka": Variant(
            path=Path("../nuitka"),
            commandline=("./partdiff",),
            do_warmup=False,
        ),
        "np_vectorize": Variant(
            path=Path("../np_vectorize"),
            commandline=("uv", "run", "--python", "cpython3.13", "main.py"),
            do_warmup=False,
        ),
        "numba": Variant(
            path=Path("../numba"),
            commandline=("uv", "run", "--python", "cpython3.10", "main.py"),
            do_warmup=True,
        ),
        "cython": Variant(
            path=Path("../cython"),
            commandline=("uv", "run", "--python", "cpython3.13", "main.py"),
            do_warmup=False,
        ),
    }
    if "nuitka" in variants.keys():
        nuitka_bin = variants["nuitka"].path / "partdiff"
        if not nuitka_bin.is_file():
            raise RuntimeError(
                "Must build nuitka variant before running benchmark "
                "or disable nuitka benchmark."
            )
    print(
        f"{'variant':<12}, i, {'method':<11}, {'runtime_internal':<16}, {'runtime_total':<13}"
    )
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(
            f"{'variant':<12}, i, {'method':<11}, {'runtime_internal':<16}, {'runtime_total':<13}\n"
        )
        for variant_name, variant in variants.items():
            cwd = variant.path
            start_i = 0 if variant.do_warmup else 1
            for i in range(start_i, NUM_REPETITIONS + 1):
                for method, args in (
                    ("Gauß-Seidel", GAUSS_SEIDEL_ARGS),
                    ("Jacobi", JACOBI_ARGS),
                ):
                    status_line(variant_name, method, i)
                    runtime_internal, runtime_total = run_and_measure(
                        variant.commandline,
                        args,
                        cwd,
                    )
                    clear_status_line()
                    line = format_row(
                        variant_name,
                        i,
                        method,
                        runtime_internal,
                        runtime_total,
                    )
                    print(line)
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
