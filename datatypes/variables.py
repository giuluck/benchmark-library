from abc import ABC
from dataclasses import dataclass, field

from datatypes.values import Value, NumericValue, CategoricalValue, PositiveValue, NegativeValue, ThresholdValue, \
    BinaryValue, CustomValue


@dataclass(repr=False, frozen=True)
class Variable(Value, ABC):
    """Abstract template for a Variable."""

    assignments: list = field(default_factory=list, init=False)
    """The list of value assignments that were taken by the variable during subsequent evaluation calls."""


@dataclass(repr=False, frozen=True)
class CustomVariable(Variable, CustomValue):
    """A variable with custom domain and dtype."""


@dataclass(repr=False, frozen=True)
class NumericVariable(Variable, NumericValue):
    """A variable with numeric type and interval-based domain."""


@dataclass(repr=False, frozen=True)
class CategoricalVariable(Variable, CategoricalValue):
    """A variable with custom data type and categorical domain."""


@dataclass(repr=False, frozen=True)
class PositiveVariable(Variable, PositiveValue):
    """A variable with numeric type which can assume positive values only."""


@dataclass(repr=False, frozen=True)
class NegativeVariable(Variable, NegativeValue):
    """A variable with numeric type which can assume negative values only."""


@dataclass(repr=False, frozen=True)
class ThresholdVariable(Variable, ThresholdValue):
    """A variable with numeric type which must be smaller than a certain threshold in absolute value."""


@dataclass(repr=False, frozen=True)
class BinaryVariable(Variable, BinaryValue):
    """A variable which can only assume value 0 or 1."""
