from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from datatypes.values import Value, NumericValue, CategoricalValue, PositiveValue, NegativeValue, ThresholdValue, \
    BinaryValue, CustomValue
from utils.strings import stringify


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class Parameter(Value, ABC):
    """Abstract template for a Parameter."""

    default: Any = field(kw_only=True)
    """The default value of the parameter."""

    def __post_init__(self):
        name, dtype, in_domain, domain, default = self.name, self.dtype, self.in_domain, self.domain, self.default
        assert isinstance(default, dtype), f"'{name}' should be of type {stringify(dtype)}, got " \
                                           f"default value {stringify(default)} of type {stringify(type(default))}"
        assert in_domain(default), f"'{name}' domain is {domain}, got default value {default}"


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class CustomParameter(Parameter, CustomValue):
    """A parameter with custom domain and dtype."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class NumericParameter(Parameter, NumericValue):
    """A parameter with numeric type and interval-based domain."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class CategoricalParameter(Parameter, CategoricalValue):
    """A parameter with custom data type and categorical domain."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class PositiveParameter(Parameter, PositiveValue):
    """A parameter with numeric type which can assume positive values only."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class NegativeParameter(Parameter, NegativeValue):
    """A parameter with numeric type which can assume negative values only."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class ThresholdParameter(Parameter, ThresholdValue):
    """A parameter with numeric type which must be smaller than a certain threshold in absolute value."""


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class BinaryParameter(Parameter, BinaryValue):
    """A parameter which can only assume value 0 or 1."""
