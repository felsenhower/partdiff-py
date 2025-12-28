"""
partdiff-py is a Python port of partdiff.
This is the "simple" variant.
"""

from itertools import count
from time import time

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
    TerminationCondition,
    parse_args,
)

PI = 3.14159265358979323846


def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    """Solve the Poisson equation iteratively using the Jacobi or Gauß-Seidel method.

    Args:
        arguments (CalculationArguments): The internal representation of the problem.
        options (Options): The program options.

    Returns:
        CalculationResults: The results of the calculation.
    """
    start_time = time()
    n = arguments.n
    h = arguments.h
    tensor = arguments.tensor
    stat_iteration = 0
    stat_accuracy = None
    matrix_out = tensor[0, :, :]
    matrix_in = matrix_out
    if options.method == CalculationMethod.JACOBI:
        matrix_in = tensor[1, :, :]
    if options.pert_func == PerturbationFunction.FPISIN:
        pih = PI * h
        fpisin = 0.25 * (2.0 * PI * PI) * h * h
    for stat_iteration in count(start=1):
        maxresiduum = 0.0
        for i in range(1, n):
            if options.pert_func == PerturbationFunction.FPISIN:
                fpisin_i = fpisin * sin(pih * float(i))
            for j in range(1, n):
                star = 0.25 * (
                    matrix_in[i - 1, j]
                    + matrix_in[i, j - 1]
                    + matrix_in[i, j + 1]
                    + matrix_in[i + 1, j]
                )
                if options.pert_func == PerturbationFunction.FPISIN:
                    star += fpisin_i * sin(pih * float(j))
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


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(options, results)


if __name__ == "__main__":
    main()
