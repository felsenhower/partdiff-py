from time import time

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
    finished = False
    while not finished:
        stat_iteration += 1
        if stat_iteration == options.term_iteration:
            finished = True
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
                if options.termination == TerminationCondition.ACCURACY or finished:
                    residuum = abs(matrix_in[i, j] - star)
                    maxresiduum = max(maxresiduum, residuum)
                matrix_out[i, j] = star
        stat_accuracy = maxresiduum
        matrix_in, matrix_out = matrix_out, matrix_in
        if options.termination == TerminationCondition.ACCURACY:
            if maxresiduum < options.term_accuracy:
                finished = True
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
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
