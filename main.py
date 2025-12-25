from parse_args import (
    parse_args,
    Options,
    CalculationMethod,
    PerturbationFunction,
    TerminationCondition,
)
from dataclasses import dataclass
import numpy as np
import sys
from math import sin
from time import time
from pympler import asizeof


@dataclass(frozen=True)
class CalculationArguments:
    N: int
    num_matrices: int
    h: float
    M: np.ndarray


@dataclass(frozen=True)
class CalculationResults:
    m: int
    stat_iteration: int
    stat_accuracy: float
    duration: float


def init_arguments(options: Options) -> CalculationArguments:
    N = (options.interlines * 8) + 9 - 1
    num_matrices = 2 if options.method == CalculationMethod.JACOBI else 1
    h = 1.0 / N
    tensor_shape = (num_matrices, N + 1, N + 1)
    M = np.zeros(tensor_shape, dtype=np.float64)
    if options.pert_func == PerturbationFunction.F0:
        for g in range(num_matrices):
            for i in range(N + 1):
                c1 = 1.0 - (h * i)
                c2 = h * i
                M[g, i, 0] = c1
                M[g, i, N] = c2
                M[g, 0, i] = c1
                M[g, N, i] = c2
            M[g, N, 0] = 0.0
            M[g, 0, N] = 0.0
    return CalculationArguments(N, num_matrices, h, M)


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    start_time = time()
    N = arguments.N
    h = arguments.h
    M = arguments.M
    stat_iteration = 0
    stat_accuracy = None
    m1, m2 = (0, 1) if options.method == CalculationMethod.JACOBI else (0, 0)
    pi = 3.14159265358979323846
    pih = pi * h
    fpisin = 0.25 * (2.0 * pi * pi) * h * h
    finished = False
    while not finished:
        stat_iteration += 1
        if stat_iteration == options.term_iteration:
            finished = True
        maxresiduum = 0
        for i in range(1, N):
            fpisin_i = fpisin * sin(pih * i)
            for j in range(1, N):
                star = 0.25 * (
                    M[m2, i - 1, j]
                    + M[m2, i, j - 1]
                    + M[m2, i, j + 1]
                    + M[m2, i + 1, j]
                )
                if options.pert_func == PerturbationFunction.FPISIN:
                    star += fpisin_i * sin(pih * j)
                if options.termination == TerminationCondition.ACCURACY or finished:
                    residuum = abs(M[m2, i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                M[m1, i, j] = star
        stat_accuracy = maxresiduum
        (m1, m2) = (m2, m1)
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                finished = True
    end_time = time()
    duration = end_time - start_time
    return CalculationResults(m2, stat_iteration, stat_accuracy, duration)


def calculate_memory_usage(
    arguments: CalculationArguments, options: Options, results: CalculationResults
) -> float:
    memory_usage = 0
    for o in (arguments, options, results):
        memory_usage += asizeof.asizeof(o)
    memory_usage = memory_usage / 1024.0 / 1024.0
    return memory_usage


def display_statistics(
    arguments: CalculationArguments, options: Options, results: CalculationResults
) -> None:
    N = arguments.N
    duration = results.duration
    memory_usage = calculate_memory_usage(arguments, options, results)
    print(f"Calculation time:       {duration:0.6f} s")
    print(f"Memory usage:           {memory_usage:0.6f} MiB")
    print(f"Calculation method:     {options.method}")
    print(f"Interlines:             {options.interlines}")
    print(f"Perturbation function:  {options.pert_func}")
    print(f"Termination:            {options.termination}")
    print(f"Number of iterations:   {results.stat_iteration}")
    print(f"Residuum:               {results.stat_accuracy:0.6e}")
    print("")


def display_matrix(
    arguments: CalculationArguments, options: Options, results: CalculationResults
) -> None:
    interlines = options.interlines
    M = arguments.M
    m = results.m
    print("Matrix:")
    for y in range(9):
        for x in range(9):
            elem = M[m, y * (interlines + 1), x * (interlines + 1)]
            print(f" {elem:0.4f}", end="")
        print("")


def check_float_info() -> None:
    float_info = sys.float_info
    assert (float_info.max_exp, float_info.mant_dig) == (1024, 53), (
        "This application does only work on platforms where built-in float is IEEE 754 binary64, e.g. CPython.",
        float_info,
    )


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
