# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True

from time import time

from libc.math cimport fabs

import numpy as np
cimport numpy as np

from partdiff_common.parse_args import (
    Options,
    CalculationMethod,
    TerminationCondition,
)

from partdiff_common import (
    CalculationArguments,
    CalculationResults,
)

cdef int METH_GAUSS_SEIDEL = CalculationMethod.GAUSS_SEIDEL.value
cdef int METH_JACOBI = CalculationMethod.JACOBI.value
cdef int TERM_ACC = TerminationCondition.ACCURACY.value
cdef int TERM_ITER = TerminationCondition.ITERATIONS.value

cdef tuple calculate_inner(
    int method,
    int termination,
    int term_iteration,
    double term_accuracy,
    int n,
    double[:, :, :] tensor,
    double[:, :] perturbation_matrix,
):
    cdef int i, j
    cdef int m1, m2
    cdef double star, residuum, maxresiduum
    cdef int stat_iteration = 0
    cdef double stat_accuracy = 0.0
    (m1, m2) = (0, 1) if method == METH_JACOBI else (0, 0)
    while term_iteration > 0:
        maxresiduum = 0.0
        for i in range(1, n):
            for j in range(1, n):
                star = 0.25 * (
                    tensor[m2, i - 1, j] + 
                    tensor[m2, i, j - 1] +
                    tensor[m2, i, j + 1] +
                    tensor[m2, i + 1, j]
                )
                star += perturbation_matrix[i, j]
                if termination == TERM_ACC or term_iteration == 1:
                    residuum = tensor[m2, i, j] - star
                    residuum = fabs(residuum)
                    if residuum > maxresiduum:
                        maxresiduum = residuum
                tensor[m1, i, j] = star
        stat_iteration += 1
        stat_accuracy = maxresiduum
        (m1, m2) = (m2, m1)
        if termination == TERM_ACC:
            if maxresiduum < term_accuracy:
                term_iteration = 0
        elif termination == TERM_ITER:
            term_iteration -= 1
    return m2, stat_iteration, stat_accuracy

def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    cdef int method = options.method.value
    cdef int termination = options.termination.value
    cdef int term_iteration = options.term_iteration
    cdef double term_accuracy = options.term_accuracy
    cdef int n = arguments.n
    cdef double[:, :, :] tensor = arguments.tensor
    cdef double[:, :] perturbation_matrix = arguments.perturbation_matrix
    start_time = time()
    m, stat_iteration, stat_accuracy = calculate_inner(
        method,
        termination,
        term_iteration,
        term_accuracy,
        n,
        tensor,
        perturbation_matrix,
    )
    end_time = time()
    duration = end_time - start_time
    final_matrix = arguments.tensor[m, :, :]
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)
