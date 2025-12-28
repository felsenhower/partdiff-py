# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True


from time import time
from itertools import count

from partdiff_common.parse_args import (
    Options,
    CalculationMethod,
    TerminationCondition,
)

from partdiff_common import (
    CalculationArguments,
    CalculationResults,
)


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    start_time = time()
    n = arguments.n
    tensor = arguments.tensor
    perturbation_matrix = arguments.perturbation_matrix
    stat_iteration = 0
    stat_accuracy = None
    matrix_out = tensor[0, :, :]
    matrix_in = matrix_out
    if options.method == CalculationMethod.JACOBI:
        matrix_in = tensor[1, :, :]
    for stat_iteration in count(start=1):
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
                    options.termination == TerminationCondition.ACCURACY
                    or stat_iteration == options.term_iteration
                ):
                    residuum = abs(matrix_in[i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                matrix_out[i, j] = star
        stat_accuracy = maxresiduum
        matrix_in, matrix_out = matrix_out, matrix_in
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
