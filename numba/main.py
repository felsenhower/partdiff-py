from time import time
from numba import njit
import numpy as np

from partdiff_common.parse_args import (
    parse_args,
    Options,
    CalculationMethod,
    PerturbationFunction,
    TerminationCondition,
    TermIterations,
    TermAccuracy,
)

from partdiff_common import (
    CalculationArguments,
    CalculationResults,
    check_float_info,
    init_arguments,
    display_statistics,
    display_matrix,
)


@njit
def calculate_iterate(
    method: CalculationMethod,
    termination: TerminationCondition,
    term_iteration: TermIterations,
    term_accuracy: TermAccuracy,
    n: int,
    tensor: np.ndarray,
    perturbation_matrix: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    stat_iteration = 0
    stat_accuracy = 0.0
    matrix_out = tensor[0, :, :]
    matrix_in = matrix_out
    if method == CalculationMethod.JACOBI:
        matrix_in = tensor[1, :, :]
    while True:
        stat_iteration += 1
        maxresiduum = 0.0
        for i in range(1, n):
            for j in range(1, n):
                star = 0.25 * (
                    matrix_in[i - 1, j]
                    + matrix_in[i, j - 1]
                    + matrix_in[i, j + 1]
                    + matrix_in[i + 1, j]
                )
                star += perturbation_matrix[i, j]
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
    start_time = time()
    final_matrix, stat_iteration, stat_accuracy = calculate_iterate(
        options.method,
        options.termination,
        options.term_iteration,
        options.term_accuracy,
        arguments.n,
        arguments.tensor,
        arguments.perturbation_matrix,
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
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
