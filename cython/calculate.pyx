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

cdef int METH_GAUSS_SEIDEL = 1
cdef int METH_JACOBI = 2
cdef int TERM_ACC = 1
cdef int TERM_ITER = 2

cdef inline double tensor_get(double* tensor, int N, int m, int i, int j) noexcept nogil:
    return tensor[m * (N+1) * (N+1) + i * (N+1) + j]

cdef inline void tensor_set(double* tensor, int N, int m, int i, int j, double value) noexcept nogil:
    tensor[m * (N+1) * (N+1) + i * (N+1) + j] = value

cdef inline double matrix_get(double* matrix, int N, int i, int j) noexcept nogil:
    return matrix[i * (N+1) + j]

cdef tuple calculate_inner(
    int N,
    int term_iteration,
    double* tensor,
    double* perturbation_matrix,
    int method,
    int termination,
    double term_accuracy
):
    cdef int m1, m2
    cdef int i, j
    cdef double star, residuum, maxresiduum
    cdef int stat_iteration = 0
    cdef double stat_accuracy = 0.0
    cdef int temp
    if method == METH_JACOBI:
        m1 = 0
        m2 = 1
    else:
        m1 = 0
        m2 = 0
    while term_iteration > 0:
        maxresiduum = 0.0
        for i in range(1, N):
            for j in range(1, N):
                star = 0.25 * (
                    tensor_get(tensor, N, m2, i-1, j) +
                    tensor_get(tensor, N, m2, i, j-1) +
                    tensor_get(tensor, N, m2, i, j+1) +
                    tensor_get(tensor, N, m2, i+1, j)
                )
                star += matrix_get(perturbation_matrix, N, i, j)
                if termination == TERM_ACC or term_iteration == 1:
                    residuum = tensor_get(tensor, N, m2, i, j) - star
                    residuum = fabs(residuum)
                    if residuum > maxresiduum:
                        maxresiduum = residuum
                tensor_set(tensor, N, m1, i, j, star)
        stat_iteration += 1
        stat_accuracy = maxresiduum
        temp = m1
        m1 = m2
        m2 = temp
        if termination == TERM_ACC:
            if maxresiduum < term_accuracy:
                term_iteration = 0
        elif termination == TERM_ITER:
            term_iteration -= 1
    return m2, stat_iteration, stat_accuracy

def calculate_np(
    int N,
    int term_iteration,
    np.ndarray[np.float64_t, ndim=3, mode="c"] tensor,
    np.ndarray[np.float64_t, ndim=2, mode="c"] perturbation_matrix,
    int method,
    int termination,
    double term_accuracy
):
    if not (1 <= tensor.shape[0] <= 2) or tensor.shape[1] != N+1 or tensor.shape[2] != N+1:
        raise ValueError("tensor must have shape (2, N+1, N+1)")
    if perturbation_matrix.shape[0] != N+1 or perturbation_matrix.shape[1] != N+1:
        raise ValueError("perturbation_matrix must have shape (N+1, N+1)")
    cdef double* tensor_ptr = <double*> tensor.data
    cdef double* matrix_ptr = <double*> perturbation_matrix.data
    cdef int m
    cdef int stat_iteration
    cdef double stat_accuracy
    m, stat_iteration, stat_accuracy = calculate_inner(
        N,
        term_iteration,
        tensor_ptr,
        matrix_ptr,
        method,
        termination,
        term_accuracy
    )
    return m, stat_iteration, stat_accuracy

def calculate(arguments: CalculationArguments, options: Options) -> CalculationResults:
    start_time = time()
    m, stat_iteration, stat_accuracy = calculate_np(
        arguments.n,
        options.term_iteration,
        arguments.tensor,
        arguments.perturbation_matrix,
        options.method.value,
        options.termination.value,
        options.term_accuracy,
    )
    end_time = time()
    duration = end_time - start_time
    final_matrix = arguments.tensor[m, :, :]
    return CalculationResults(final_matrix, stat_iteration, stat_accuracy, duration)
