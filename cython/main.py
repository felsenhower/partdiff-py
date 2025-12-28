"""
partdiff-py is a Python port of partdiff.
This is the "cython" variant. It uses Cython to translate the calculate method to C.
"""

from calculate import calculate
from partdiff_common import (
    check_float_info,
    display_matrix,
    display_statistics,
    init_arguments,
)
from partdiff_common.parse_args import parse_args


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(options, results)


if __name__ == "__main__":
    main()
