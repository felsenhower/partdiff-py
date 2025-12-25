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


def calculate_jacobi(
    arguments: CalculationArguments, options: Options
) -> CalculationResults:
    start_time = time()
    n = arguments.n
    tensor = arguments.tensor
    perturbation_matrix = arguments.perturbation_matrix
    stat_iteration = 0
    stat_accuracy = None
    m1, m2 = (0, 1) if options.method == CalculationMethod.JACOBI else (0, 0)
    finished = False
    while not finished:
        stat_iteration += 1
        if stat_iteration == options.term_iteration:
            finished = True
        maxresiduum = 0.0
        for i in range(1, n):
            for j in range(1, n):
                star = 0.25 * (
                    tensor[m2, i - 1, j]
                    + tensor[m2, i, j - 1]
                    + tensor[m2, i, j + 1]
                    + tensor[m2, i + 1, j]
                )
                star += perturbation_matrix[i, j]
                if options.termination == TerminationCondition.ACCURACY or finished:
                    residuum = abs(tensor[m2, i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                tensor[m1, i, j] = star
        stat_accuracy = maxresiduum
        (m1, m2) = (m2, m1)
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                finished = True
    end_time = time()
    duration = end_time - start_time
    final_matrix = tensor[m2, :, :]
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)


def calculate_gauss_seidel(
    arguments: CalculationArguments, options: Options
) -> CalculationResults:
    start_time = time()
    n = arguments.n
    tensor = arguments.tensor
    perturbation_matrix = arguments.perturbation_matrix
    stat_iteration = 0
    stat_accuracy = None
    m1, m2 = (0, 1) if options.method == CalculationMethod.JACOBI else (0, 0)
    finished = False
    while not finished:
        stat_iteration += 1
        if stat_iteration == options.term_iteration:
            finished = True
        maxresiduum = 0.0
        for i in range(1, n):
            for j in range(1, n):
                star = 0.25 * (
                    tensor[m2, i - 1, j]
                    + tensor[m2, i, j - 1]
                    + tensor[m2, i, j + 1]
                    + tensor[m2, i + 1, j]
                )
                star += perturbation_matrix[i, j]
                if options.termination == TerminationCondition.ACCURACY or finished:
                    residuum = abs(tensor[m2, i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                tensor[m1, i, j] = star
        stat_accuracy = maxresiduum
        (m1, m2) = (m2, m1)
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                finished = True
    end_time = time()
    duration = end_time - start_time
    final_matrix = tensor[m2, :, :]
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    match options.method:
        case CalculationMethod.JACOBI:
            return calculate_jacobi(arguments, options)
        case CalculationMethod.GAUSS_SEIDEL:
            return calculate_gauss_seidel(arguments, options)


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


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
