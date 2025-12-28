"""
partdiff-py is a Python port of partdiff.
This is the "numba" variant. It uses numba to just-in-time compile the calculation.
"""

from time import time

import numpy as np
from numba import njit
from numpy import sin
from partdiff_common import (
    CalculationArguments,
    CalculationResults,
    check_float_info,
    display_matrix,
    display_statistics,
    init_arguments,
)
from partdiff_common.parse_args import (
    CalculationMethod,
    Options,
    PerturbationFunction,
    TermAccuracy,
    TerminationCondition,
    TermIterations,
    parse_args,
)

PI = 3.14159265358979323846


@njit
def calculate_iterate(
    method: CalculationMethod,
    pert_func: PerturbationFunction,
    termination: TerminationCondition,
    term_iteration: TermIterations,
    term_accuracy: TermAccuracy,
    n: int,
    h: float,
    tensor: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """The inner calculation part of the calculate method which is just-in-time compiled
    with numba here.

    Args:
        method (CalculationMethod): The method (Gauß-Seidel or Jacobi).
        pert_func (PerturbationFunction): The perturbation function.
        termination (TerminationCondition): Termination (Iterations or Accuracy).
        term_iteration (TermIterations): Max iterations.
        term_accuracy (TermAccuracy): Min accuracy.
        n (int): Problem size (matrix is (n+1)*(n+1)).
        tensor (np.ndarray): The problem matrices.

    Returns:
        tuple[np.ndarray, int, float]: A tuple containing the final matrix,
            actual iterations performed, and residuum reached.
    """
    stat_iteration = 0
    stat_accuracy = 0.0
    matrix_out = tensor[0, :, :]
    matrix_in = matrix_out
    if method == CalculationMethod.JACOBI:
        matrix_in = tensor[1, :, :]
    if pert_func == PerturbationFunction.FPISIN:
        pih = PI * h
        fpisin = 0.25 * (2.0 * PI * PI) * h * h
    while True:
        stat_iteration += 1
        maxresiduum = 0.0
        for i in range(1, n):
            if pert_func == PerturbationFunction.FPISIN:
                fpisin_i = fpisin * sin(pih * float(i))
            for j in range(1, n):
                star = 0.25 * (
                    matrix_in[i - 1, j]
                    + matrix_in[i, j - 1]
                    + matrix_in[i, j + 1]
                    + matrix_in[i + 1, j]
                )
                if pert_func == PerturbationFunction.FPISIN:
                    star += fpisin_i * sin(pih * float(j))
                if (
                    termination == TerminationCondition.ACCURACY
                    or stat_iteration == term_iteration
                ):
                    residuum = abs(matrix_in[i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                matrix_out[i, j] = star
        stat_accuracy = maxresiduum
        matrix_in, matrix_out = matrix_out, matrix_in
        if termination == TerminationCondition.ACCURACY:
            if maxresiduum < term_accuracy:
                break
        else:
            if stat_iteration == term_iteration:
                break
    final_matrix = matrix_in
    return final_matrix, stat_iteration, stat_accuracy


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    """Solve the Poisson equation iteratively using the Jacobi or Gauß-Seidel method.

    Args:
        arguments (CalculationArguments): The internal representation of the problem.
        options (Options): The program options.

    Returns:
        CalculationResults: The results of the calculation.
    """
    start_time = time()
    final_matrix, stat_iteration, stat_accuracy = calculate_iterate(
        options.method,
        options.pert_func,
        options.termination,
        options.term_iteration,
        options.term_accuracy,
        arguments.n,
        arguments.h,
        arguments.tensor,
    )
    end_time = time()
    duration = end_time - start_time
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(options, results)


if __name__ == "__main__":
    main()
