from time import time
import numpy as np
from itertools import count

from partdiff_common.parse_args import (
    parse_args,
    Options,
    CalculationMethod,
    PerturbationFunction,
    TerminationCondition,
)

from partdiff_common import (
    CalculationArguments,
    CalculationResults,
    check_float_info,
    init_arguments,
    display_statistics,
    display_matrix,
)


def calculate_jacobi(
    arguments: CalculationArguments, options: Options
) -> CalculationResults:
    start_time = time()
    n = arguments.n
    tensor = arguments.tensor
    perturbation_matrix = arguments.perturbation_matrix
    stat_accuracy = None
    matrix_out = tensor[0, :, :]
    matrix_in = tensor[1, :, :]
    for stat_iteration in count(start=1):
        maxresiduum = 0.0
        center = matrix_in[1:n, 1:n]
        north = matrix_in[0 : n - 1, 1:n]
        west = matrix_in[1:n, 0 : n - 1]
        east = matrix_in[1:n, 2 : n + 1]
        south = matrix_in[2 : n + 1, 1:n]
        pert = perturbation_matrix[1:n, 1:n]
        new = matrix_out[1:n, 1:n]
        new[:] = 0.25 * (north + west + east + south) + pert
        if (
            options.termination == TerminationCondition.ACCURACY
            or stat_iteration == options.term_iteration
        ):
            maxresiduum = np.abs(center - new).max()
        stat_accuracy = maxresiduum
        matrix_out, matrix_in = matrix_in, matrix_out
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                break
        else:
            if stat_iteration == options.term_iteration:
                break
    end_time = time()
    duration = end_time - start_time
    final_matrix = matrix_in
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
    matrix = tensor[0, :, :]
    for stat_iteration in count(start=1):
        maxresiduum = 0.0
        for i in range(1, n):
            for j in range(1, n):
                star = 0.25 * (
                    matrix[i - 1, j]
                    + matrix[i, j - 1]
                    + matrix[i, j + 1]
                    + matrix[i + 1, j]
                )
                star += perturbation_matrix[i, j]
                if (
                    options.termination == TerminationCondition.ACCURACY
                    or stat_iteration == options.term_iteration
                ):
                    residuum = abs(matrix[i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                matrix[i, j] = star
        stat_accuracy = maxresiduum
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                break
        else:
            if stat_iteration == options.term_iteration:
                break
    end_time = time()
    duration = end_time - start_time
    final_matrix = matrix
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    match options.method:
        case CalculationMethod.JACOBI:
            return calculate_jacobi(arguments, options)
        case CalculationMethod.GAUSS_SEIDEL:
            return calculate_gauss_seidel(arguments, options)


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
