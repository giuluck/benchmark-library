from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from datatypes.datatype import DataType


@dataclass(repr=False, frozen=True)
class Value(DataType, ABC):
    """Abstract class for data types with a value (variables and parameters)."""

    @property
    @abstractmethod
    def dtype(self) -> type:
        """The expected instance type."""
        pass

    @property
    @abstractmethod
    def domain(self) -> str:
        """A textual description of the instance domain."""
        pass

    @abstractmethod
    def in_domain(self, v) -> bool:
        """Checks if the given value is in the domain of the instance.

        :param v:
            The given value for the instance.

        :return:
            A boolean value which indicates whether the assigned value falls or not into the domain of the instance.
        """
        pass


@dataclass(repr=False, frozen=True)
class CustomValue(Value):
    """A value object with custom domain and dtype."""

    dtype: type = field(default=object, kw_only=True)
    """The instance type; default: object."""

    domain: str = field(default='all', kw_only=True)
    """A textual description of the domain; default: 'all'."""

    in_domain: Callable[[Any], bool] = field(default=lambda v: True, kw_only=True)
    """A function f(x) -> bool which defines whether the variable domain; default: f(x) -> True."""


@dataclass(repr=False, frozen=True)
class NumericValue(Value):
    """A value object with numeric type and interval-based domain."""

    integer: bool = field(default=False, kw_only=True)
    """Whether the value is expected to be an integer or a real number; default: False."""

    internal: bool = field(default=True, kw_only=True)
    """Whether the domain is internal to the bounds (i.e., [lb, ub]) or external (i.e., < lb or > ub); default: True."""

    lb: Optional[float] = field(default=None, kw_only=True)
    """The lower bound of the interval or None for no lower bound; default: None."""

    ub: Optional[float] = field(default=None, kw_only=True)
    """The upper bound of the interval or None for no upper bound; default: None."""

    strict_lb: bool = field(default=False, kw_only=True)
    """Whether the lower bound is strict or not; default: False."""

    strict_ub: bool = field(default=False, kw_only=True)
    """Whether the upper bound is strict or not; default: False."""

    def __post_init__(self):
        lb, ub = self.lb, self.ub
        assert lb is None or ub is None or ub >= lb, f"'ub' must be greater or equal to 'lb', got {ub} < {lb}"

    @property
    def dtype(self) -> type:
        return int if self.integer else float

    @property
    def domain(self) -> str:
        return self._get_domain()

    def in_domain(self, v) -> bool:
        return self._get_domain(v)

    def _get_domain(self, v: Optional = None) -> bool | Optional[str]:
        description = v is None
        # handle all the scenarios (half-bounded vs. double bounded; strict vs. non-strict; internal vs. external)
        match (self.lb, self.ub, self.strict_lb, self.strict_ub, self.internal):
            case (None, None, _, _, _):
                # no bounds
                return None if description else True
            case (lb, None, True, _, True):
                # internal strict lower bound only
                return f"]{lb}, +inf[" if description else v > lb
            case (lb, None, False, _, True):
                # internal non-strict lower bound only
                return f"[{lb}, +inf[" if description else v >= lb
            case (None, ub, _, True, True):
                # internal strict upper bound only
                return f"]-inf, {ub}[" if description else v < ub
            case (None, ub, _, False, True):
                # internal non-strict upper bound only
                return f"]-inf, {ub}]" if description else v <= ub
            case (lb, ub, True, True, True):
                # internal strict lb and ub
                return f"]{lb}, {ub}[" if description else lb < v < ub
            case (lb, ub, False, True, True):
                # internal non-strict lb strict ub
                return f"[{lb}, {ub}[" if description else lb <= v < ub
            case (lb, ub, True, False, True):
                # internal strict lb non-strict ub
                return f"]{lb}, {ub}]" if description else lb < v <= ub
            case (lb, ub, False, False, True):
                # internal non-strict lb and ub
                return f"[{lb}, {ub}]" if description else lb <= v <= ub
            case (lb, None, True, _, False):
                # external strict lower bound only
                return f"out of [{lb}, +inf[" if description else v < lb
            case (lb, None, False, _, False):
                # external non-strict lower bound only
                return f"out of ]{lb}, +inf[" if description else v <= lb
            case (None, ub, _, True, False):
                # external strict upper bound only
                return f"out of ]-inf, {ub}]" if description else v > ub
            case (None, ub, _, False, False):
                # external non-strict upper bound only
                return f"out of ]-inf, {ub}[" if description else v >= ub
            case (lb, ub, True, True, False):
                # external strict lb and ub
                return f"out of [{lb}, {ub}]" if description else v < lb or v > ub
            case (lb, ub, False, True, False):
                # external non-strict lb strict ub
                return f"out of ]{lb}, {ub}]" if description else v <= lb or v > ub
            case (lb, ub, True, False, False):
                # external strict lb non-strict ub
                return f"out of [{lb}, {ub}[" if description else v < lb or v >= ub
            case (lb, ub, False, False, False):
                # external non-strict lb and ub
                return f"out of ]{lb}, {ub}[" if description else v <= lb or v >= ub
            case _:
                raise RuntimeError("Something went wrong during constraint checks")


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class CategoricalValue(Value):
    """A Value object with custom data type and categorical domain."""

    categories: set = field(kw_only=True)
    """The discrete set of possible values defining the instance domain."""

    dtype: type = field(default=object, kw_only=True)
    """The expected instance type; default: object."""

    @property
    def domain(self) -> str:
        return str(self.categories)

    def in_domain(self, v) -> bool:
        return v in self.categories


# noinspection PyDataclass,PyRedeclaration
@dataclass(repr=False, frozen=True)
class PositiveValue(NumericValue):
    """A value object with numeric type which can assume positive values only."""

    strict: bool = field(default=False, kw_only=True)
    """Whether the value must be strictly positive or it can have value zero; default = False."""

    # non-init parameters
    internal: bool = field(default=True, init=False)
    lb: Optional[float] = field(default=0.0, init=False)
    ub: Optional[float] = field(default=None, init=False)
    strict_ub: bool = field(default=True, init=False)

    # redeclare 'strict_lb' as a property to match 'strict'
    strict_lb: bool = field(init=False)

    @property
    def strict_lb(self) -> bool:
        return self.strict

    @strict_lb.setter
    def strict_lb(self, value: bool):
        pass


# noinspection PyDataclass,PyRedeclaration
@dataclass(repr=False, frozen=True)
class NegativeValue(NumericValue):
    """A value object with numeric type which can assume negative values only."""

    strict: bool = field(default=False, kw_only=True)
    """Whether the value must be strictly negative or it can have value zero; default = False."""

    # non-init parameters
    internal: bool = field(default=True, init=False)
    lb: Optional[float] = field(default=None, init=False)
    ub: Optional[float] = field(default=0.0, init=False)
    strict_ub: bool = field(default=False, init=False)

    # redeclare 'strict_ub' as a property to match 'strict'
    strict_lb: bool = field(init=False)

    @property
    def strict_ub(self) -> bool:
        return self.strict

    @strict_ub.setter
    def strict_ub(self, value: bool):
        pass


# noinspection PyDataclass,PyRedeclaration
@dataclass(repr=False, frozen=True)
class ThresholdValue(NumericValue):
    """A value object with numeric type which must be smaller than a certain threshold in absolute value."""

    threshold: float = field(kw_only=True)
    """A positive value representing the threshold."""

    strict: bool = field(default=False, kw_only=True)
    """Whether the value must be strictly smaller than the threshold or not; default = False."""

    # non-init parameters
    internal: bool = field(default=True, init=False)

    def __post_init__(self):
        assert self.threshold > 0, f"threshold should be a positive value, got {self.threshold}"

    # redeclare bounds and strict as properties to match new parameters
    lb: Optional[float] = field(init=False)
    ub: Optional[float] = field(init=False)
    strict_lb: bool = field(init=False)
    strict_ub: bool = field(init=False)

    @property
    def lb(self) -> float:
        return -self.threshold

    @lb.setter
    def lb(self, value: float):
        pass

    @property
    def ub(self) -> float:
        return self.threshold

    @ub.setter
    def ub(self, value: float):
        pass

    @property
    def strict_lb(self) -> bool:
        return self.strict

    @strict_lb.setter
    def strict_lb(self, value: bool):
        pass

    @property
    def strict_ub(self) -> bool:
        return self.strict

    @strict_ub.setter
    def strict_ub(self, value: bool):
        pass


# noinspection PyDataclass,PyRedeclaration
@dataclass(repr=False, frozen=True)
class BinaryValue(CategoricalValue):
    """A value object which can only assume value 0 or 1."""

    # non-init parameters
    categories: tuple = field(default_factory=lambda: {0, 1}, init=False)
    dtype: type = field(default=int, init=False)
