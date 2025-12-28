"""
This module contains the shared parts of partdiff, e.g. all the stuff for
initializing matrices or displaying statistics.
"""

import sys
from dataclasses import dataclass

import numpy as np
from pympler import asizeof

from partdiff_common.parse_args import (
    CalculationMethod,
    Options,
    PerturbationFunction,
)


@dataclass(frozen=True)
class CalculationArguments:
    """This class contains the internal representation of the problem, i.e. the
    initialized matrices. All matrices have the size (n+1)*(n+1).
    The tensor may be 1*(n+1)*(n+1) or 2*(n+1)*(n+1), depending on the method.
    """
    n: int
    h: float
    tensor: np.ndarray


@dataclass(frozen=True)
class CalculationResults:
    """This class contains the final results of the calculation."""

    final_matrix: np.ndarray
    stat_iteration: int
    stat_accuracy: float
    duration: float


def init_arguments(options: Options) -> CalculationArguments:
    """Init the CalculationArguments.

    Args:
        options (Options): The program options.

    Returns:
        CalculationArguments: The initialized problem representation.
    """
    n = (options.interlines * 8) + 9 - 1
    num_matrices = 2 if options.method == CalculationMethod.JACOBI else 1
    h = 1.0 / n
    tensor_shape = (num_matrices, n + 1, n + 1)
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
    return CalculationArguments(n, h, tensor)


def calculate_memory_usage(
    arguments: CalculationArguments, options: Options, results: CalculationResults
) -> float:
    """Calculate the total memory usage.

    Args:
        arguments (CalculationArguments): The calculation arguments.
        options (Options): The program options.
        results (CalculationResults): The calculation results.

    Returns:
        float: The memory usage in MiB.
    """
    memory_usage = 0
    for o in (arguments, options, results):
        memory_usage += asizeof.asizeof(o)
    memory_usage = memory_usage / 1024.0 / 1024.0
    return memory_usage


def display_statistics(
    arguments: CalculationArguments, options: Options, results: CalculationResults
) -> None:
    """Display statistics about the calculation.

    Args:
        arguments (CalculationArguments): The calculation arguments.
        options (Options): The program options.
        results (CalculationResults): The calculation results.
    """
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


def display_matrix(options: Options, results: CalculationResults) -> None:
    """Display the final matrix in a 9x9 format.

    Args:
        options (Options): The program options.
        results (CalculationResults): The calculation results.
    """
    interlines = options.interlines
    final_matrix = results.final_matrix
    print("Matrix:")
    for y in range(9):
        for x in range(9):
            elem = final_matrix[y * (interlines + 1), x * (interlines + 1)]
            print(f" {elem:0.4f}", end="")
        print("")


def check_float_info() -> None:
    """Check that we're running a platform where Python's builtin float is double."""
    float_info = sys.float_info
    assert (float_info.max_exp, float_info.mant_dig) == (1024, 53), (
        "This application does only work on platforms where built-in float is IEEE 754 binary64, e.g. CPython.",
        float_info,
    )
