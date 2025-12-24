import numpy as np
import argparse
from enum import Enum
from pydantic.dataclasses import dataclass
from pydantic import TypeAdapter, ValidationError, ConfigDict
from typing import Annotated
import typing
from annotated_types import Ge, Le


class LabeledIntEnum(Enum):
    def __new__(cls, value: int, label: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        return obj

    def __str__(self) -> str:
        return self.label


class CalculationMethod(LabeledIntEnum):
    GAUSS_SEIDEL = (1, "Gauß-Seidel")
    JACOBI = (2, "Jacobi")


class PerturbationFunction(LabeledIntEnum):
    F0 = (1, "f(x,y) = 0")
    FPISIN = (2, "f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)")


class TerminationCondition(LabeledIntEnum):
    ACCURACY = (1, "Required accuracy")
    ITERATIONS = (2, "Number of iterations")


def enum_parser(enum_cls: type):
    def parse_enum(value: str) -> enum_cls:
        as_int = int(value)
        return enum_cls(as_int)

    return parse_enum


NumThreads = Annotated[int, Ge(1), Le(1024)]
NumInterlines = Annotated[int, Ge(0), Le(100_000)]
TermIterations = Annotated[int, Ge(1), Le(200_000)]
TermAccuracy = Annotated[np.float64, Ge(1e-20), Le(1e-4)]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Options:
    num_threads: NumThreads
    method: CalculationMethod
    interlines: NumInterlines
    pert_func: PerturbationFunction
    termination: TerminationCondition
    term_iteration: TermIterations | None
    term_accuracy: TermAccuracy | None


def annotated_parser(model):
    def parse(value: str):
        try:
            adapter = TypeAdapter(model)
            result = adapter.validate_strings(value)
            return result
        except ValidationError as e:
            raise ValueError() from e

    return parse


def parse_term_acc_iter(termination: TerminationCondition, value: str):
    match termination:
        case TerminationCondition.ACCURACY:
            return (None, annotated_parser(TermAccuracy)(value))
        case TerminationCondition.ITERATIONS:
            return (annotated_parser(TermIterations)(value), None)


def annotated_range(model):
    meta = model.__metadata__
    minimum = max(ann.ge for ann in meta if isinstance(ann, Ge))
    maximum = min(ann.le for ann in meta if isinstance(ann, Le))
    return (minimum, maximum)


def enum_range(cls):
    minimum = min(choice.value for choice in cls)
    maximum = max(choice.value for choice in cls)
    return (minimum, maximum)


def type_range(cls):
    if typing.get_origin(cls) == Annotated:
        return annotated_range(cls)
    if issubclass(cls, Enum):
        return enum_range(cls)
    assert False, "Unexpected type found"


def display_range(cls, strformat="{}"):
    minimum, maximum = type_range(cls)
    return "{}..{}".format(strformat.format(minimum), strformat.format(maximum))


def help_for_enum(cls):
    return "\n".join(f"{choice.value}: {choice.label}" for choice in cls)


def indent(lines: str):
    return "\n".join(f"  {line}" for line in lines.split("\n"))


def parse_args() -> Options:
    parser = argparse.ArgumentParser(
        "partdiff", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "num_threads",
        metavar="num",
        type=annotated_parser(NumThreads),
        help=f"Number of threads ({display_range(NumThreads)}).",
    )
    parser.add_argument(
        "method",
        metavar="method",
        type=enum_parser(CalculationMethod),
        help=f"Calculation method ({display_range(CalculationMethod)}).\n{indent(help_for_enum(CalculationMethod))}",
    )
    parser.add_argument(
        "interlines",
        metavar="lines",
        type=annotated_parser(NumInterlines),
        help=f"Number of interlines ({display_range(NumInterlines)}).\n{indent('matrixsize = (interlines * 8) + 9')}",
    )
    parser.add_argument(
        "pert_func",
        metavar="func",
        type=enum_parser(PerturbationFunction),
        help=f"Perturbation function ({display_range(PerturbationFunction)}).\n{indent(help_for_enum(PerturbationFunction))}",
    )
    parser.add_argument(
        "termination",
        metavar="term",
        type=enum_parser(TerminationCondition),
        help=f"Termination condition ({display_range(TerminationCondition)}).\n{indent(help_for_enum(TerminationCondition))}",
    )
    parser.add_argument(
        "acc_iter",
        metavar="acc/iter",
        help=(
            "depending on term\n"
            + indent(
                f"accuracy: {display_range(TermAccuracy, '{:0.0e}')}\n"
                f"iterations:   {display_range(TermIterations)}"
            )
        ),
    )
    args = parser.parse_args()
    try:
        (term_iteration, term_accuracy) = parse_term_acc_iter(
            args.termination, args.acc_iter
        )
    except ValueError:
        parser.error("Invalid value for acc/iter.")
    return Options(
        num_threads=args.num_threads,
        method=args.method,
        interlines=args.interlines,
        pert_func=args.pert_func,
        termination=args.termination,
        term_iteration=term_iteration,
        term_accuracy=term_accuracy,
    )
