# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True

from time import time

from libc.math cimport fabs, sin

import numpy as np
cimport numpy as np

from partdiff_common.parse_args import (
    Options,
    CalculationMethod,
    TerminationCondition,
    PerturbationFunction,
)

from partdiff_common import (
    CalculationArguments,
    CalculationResults,
)

cdef int METHOD_GAUSS_SEIDEL = CalculationMethod.GAUSS_SEIDEL.value
cdef int METHOD_JACOBI = CalculationMethod.JACOBI.value
cdef int TERMINATION_ACCURACY = TerminationCondition.ACCURACY.value
cdef int TERMINATION_ITERATIONS = TerminationCondition.ITERATIONS.value
cdef int PERT_FUNC_F0 = PerturbationFunction.F0.value
cdef int PERT_FUNC_FPISIN = PerturbationFunction.FPISIN.value

cdef double PI = 3.14159265358979323846

cdef tuple calculate_inner(
    int method,
    int pert_func,
    int termination,
    int term_iteration,
    double term_accuracy,
    int n,
    double h,
    double[:, :, :] tensor,
):
    cdef int i, j
    cdef double star, residuum, maxresiduum, pih, fpisin, fpisin_i
    cdef int stat_iteration = 0
    cdef double stat_accuracy = 0.0
    cdef double[:, :] matrix_out = tensor[0, :, :]
    cdef double[:, :] matrix_in = matrix_out
    cdef double[:, :] final_matrix
    pih = 0.0
    fpisin = 0.0
    fpisin_i = 0.0
    if method == METHOD_JACOBI:
        matrix_in = tensor[1, :, :]
    if pert_func == PERT_FUNC_FPISIN:
        pih = PI * h
        fpisin = 0.25 * (2.0 * PI * PI) * h * h
    while True:
        stat_iteration += 1
        maxresiduum = 0.0
        for i in range(1, n):
            if pert_func == PERT_FUNC_FPISIN:
                fpisin_i = fpisin * sin(pih * i)
            for j in range(1, n):
                star = 0.25 * (
                    matrix_in[i - 1, j] + 
                    matrix_in[i, j - 1] +
                    matrix_in[i, j + 1] +
                    matrix_in[i + 1, j]
                )
                if pert_func == PERT_FUNC_FPISIN:
                    star += fpisin_i * sin(pih * j)
                if (
                    termination == TERMINATION_ACCURACY
                    or term_iteration == stat_iteration
                ):
                    residuum = fabs(matrix_in[i, j] - star)
                    if residuum > maxresiduum:
                        maxresiduum = residuum
                matrix_out[i, j] = star
        stat_accuracy = maxresiduum
        matrix_in, matrix_out = matrix_out, matrix_in
        if termination == TERMINATION_ACCURACY:
            if maxresiduum < term_accuracy:
                break
        else:
            if stat_iteration == term_iteration:
                break
    final_matrix = matrix_in
    return final_matrix, stat_iteration, stat_accuracy

def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    cdef int method = options.method.value
    cdef int pert_func = options.pert_func.value
    cdef int termination = options.termination.value
    cdef int term_iteration = options.term_iteration
    cdef double term_accuracy = options.term_accuracy
    cdef int n = arguments.n
    cdef double h = arguments.h
    cdef double[:, :, :] tensor = arguments.tensor
    start_time = time()
    final_matrix, stat_iteration, stat_accuracy = calculate_inner(
        method,
        pert_func,
        termination,
        term_iteration,
        term_accuracy,
        n,
        h,
        tensor,
    )
    end_time = time()
    duration = end_time - start_time
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)
