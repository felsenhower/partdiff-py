from partdiff_common.parse_args import parse_args

from partdiff_common import (
    check_float_info,
    init_arguments,
    display_statistics,
    display_matrix,
)

from calculate import calculate


def main() -> None:
    check_float_info()
    options = parse_args()
    arguments = init_arguments(options)
    results = calculate(arguments, options)
    display_statistics(arguments, options, results)
    display_matrix(arguments, options, results)


if __name__ == "__main__":
    main()
