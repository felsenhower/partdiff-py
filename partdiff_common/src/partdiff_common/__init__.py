from partdiff_common.parse_args import (
    Options,
    CalculationMethod,
    PerturbationFunction,
    TerminationCondition,
)
from dataclasses import dataclass
import numpy as np
import sys
from math import sin

from pympler import asizeof


@dataclass(frozen=True)
class CalculationArguments:
    n: int
    h: float
    tensor: np.ndarray
    perturbation_matrix: np.ndarray


@dataclass(frozen=True)
class CalculationResults:
    final_matrix: np.ndarray
    stat_iteration: int
    stat_accuracy: float
    duration: float


def init_arguments(options: Options) -> CalculationArguments:
    n = (options.interlines * 8) + 9 - 1
    num_matrices = 2 if options.method == CalculationMethod.JACOBI else 1
    h = 1.0 / n
    matrix_shape = (n + 1, n + 1)
    tensor_shape = (num_matrices, *matrix_shape)
    tensor = np.zeros(tensor_shape, dtype=np.float64)
    if options.pert_func == PerturbationFunction.F0:
        for g in range(num_matrices):
            for i in range(n + 1):
                c1 = 1.0 - (h * i)
                c2 = h * i
                tensor[g, i, 0] = c1
                tensor[g, i, n] = c2
                tensor[g, 0, i] = c1
                tensor[g, n, i] = c2
            tensor[g, n, 0] = 0.0
            tensor[g, 0, n] = 0.0
    perturbation_matrix = np.zeros(matrix_shape, dtype=np.float64)
    if options.pert_func == PerturbationFunction.FPISIN:
        pi = 3.14159265358979323846
        pih = pi * h
        fpisin = 0.25 * (2.0 * pi * pi) * h * h
        for i in range(1, n):
            fpisin_i = fpisin * sin(pih * i)
            for j in range(1, n):
                perturbation_matrix[i, j] = fpisin_i * sin(pih * j)
    return CalculationArguments(n, h, tensor, perturbation_matrix)


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
    memory_usage = calculate_memory_usage(arguments, options, results)
    print(f"Calculation time:       {results.duration:0.6f} s")
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
    final_matrix = results.final_matrix
    print("Matrix:")
    for y in range(9):
        for x in range(9):
            elem = final_matrix[y * (interlines + 1), x * (interlines + 1)]
            print(f" {elem:0.4f}", end="")
        print("")


def check_float_info() -> None:
    float_info = sys.float_info
    assert (float_info.max_exp, float_info.mant_dig) == (1024, 53), (
        "This application does only work on platforms where built-in float is IEEE 754 binary64, e.g. CPython.",
        float_info,
    )
