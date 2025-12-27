import argparse
import inspect
import typing
from enum import Enum
from typing import Annotated

from annotated_types import Ge, Le
from pydantic import TypeAdapter, ValidationError, Field
from pydantic.dataclasses import dataclass


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


NumThreads = Annotated[int, Field(ge=1, le=1024)]
NumInterlines = Annotated[int, Field(ge=0, le=100_000)]
TermIterations = Annotated[int, Field(ge=1, le=200_000, default=200_000)]
TermAccuracy = Annotated[float, Field(ge=1e-20, le=1e-4, default=1e-20)]


@dataclass(frozen=True)
class Options:
    num_threads: NumThreads
    method: CalculationMethod
    interlines: NumInterlines
    pert_func: PerturbationFunction
    termination: TerminationCondition
    term_iteration: TermIterations
    term_accuracy: TermAccuracy


def enum_parser(enum_cls: type):
    def parse_enum(value: str) -> enum_cls:
        as_int = int(value)
        return enum_cls(as_int)

    return parse_enum


def annotated_parser(model):
    def parse(value: str):
        try:
            adapter = TypeAdapter(model)
            result = adapter.validate_strings(value)
            return result
        except ValidationError as e:
            raise ValueError() from e

    return parse


def type_parser(cls):
    if cls is None:
        return None
    if typing.get_origin(cls) == Annotated:
        return annotated_parser(cls)
    if inspect.isclass(cls) and issubclass(cls, Enum):
        return enum_parser(cls)
    assert False, "Unexpected type found"


def parse_term_acc_iter(termination: TerminationCondition, value: str):
    match termination:
        case TerminationCondition.ACCURACY:
            return (get_default(TermIterations), annotated_parser(TermAccuracy)(value))
        case TerminationCondition.ITERATIONS:
            return (annotated_parser(TermIterations)(value), get_default(TermAccuracy))


def get_default(model):
    meta = model.__metadata__
    assert isinstance(meta, tuple)
    assert len(meta) == 1
    (field_info,) = meta
    return field_info.default


def annotated_range(model):
    meta = model.__metadata__
    assert isinstance(meta, tuple)
    assert len(meta) == 1
    (field_info,) = meta
    meta = field_info.metadata
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
    if inspect.isclass(cls) and issubclass(cls, Enum):
        return enum_range(cls)
    assert False, "Unexpected type found"


def display_range(cls, strformat="{}"):
    minimum, maximum = type_range(cls)
    return "{}..{}".format(strformat.format(minimum), strformat.format(maximum))


def help_for_enum(cls):
    return "\n".join(f"{choice.value}: {choice.label}" for choice in cls)


def parse_args() -> Options:
    def add_argument(parser, name, **kwargs):
        type = kwargs.get("type", None)
        help = kwargs.get("help", None)
        assert help is not None
        metavar = kwargs.get("metavar", name)
        extra_help = kwargs.get("extra_help", None)
        help_text = help
        if type is not None:
            help_text += f" ({display_range(type)})."
        if inspect.isclass(type) and issubclass(type, Enum):
            help_text += "\n" + help_for_enum(type)
        if extra_help is not None:
            help_text += "\n" + extra_help
        help_text = help_text.split("\n")
        help_text = [help_text[0], *[f"  {line}" for line in help_text[1:]]]
        help_text = "\n".join(help_text)
        parser.add_argument(
            name,
            metavar=metavar,
            type=type_parser(type),
            help=help_text,
        )

    parser = argparse.ArgumentParser(
        "partdiff", formatter_class=argparse.RawTextHelpFormatter
    )
    add_argument(
        parser, "num_threads", metavar="num", type=NumThreads, help="Number of threads"
    )
    add_argument(parser, "method", type=CalculationMethod, help="Calculation method")
    add_argument(
        parser,
        "interlines",
        metavar="lines",
        type=NumInterlines,
        help="Number of interlines",
        extra_help="matrixsize = (interlines * 8) + 9",
    )
    add_argument(
        parser,
        "pert_func",
        metavar="func",
        type=PerturbationFunction,
        help="Perturbation function",
    )
    add_argument(
        parser,
        "termination",
        metavar="term",
        type=TerminationCondition,
        help="Termination condition",
    )
    add_argument(
        parser,
        "acc_iter",
        metavar="acc/iter",
        help="depending on term",
        extra_help=(
            f"accuracy: {display_range(TermAccuracy, '{:0.0e}')}\n"
            f"iterations:   {display_range(TermIterations)}"
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
