from abc import ABCMeta, abstractmethod
from typing import Optional

from datatypes.datatype import DataType


class Value(DataType, metaclass=ABCMeta):
    """Interface for a Data Type with a value (variables and parameters)."""

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


class NumericValue(Value, metaclass=ABCMeta):
    """Interface for a Value object with numeric type and interval-based domain."""

    @property
    @abstractmethod
    def integer(self) -> bool:
        """Whether the value is expected to be an integer or a real number."""
        pass

    @property
    @abstractmethod
    def internal(self) -> bool:
        """Whether the domain is internal to the bounds (i.e., [lb, ub]) or external (i.e., [-inf, lb] U [ub, +inf])."""
        pass

    @property
    @abstractmethod
    def lb(self) -> Optional[float]:
        """The lower bound of the interval or None for no lower bound."""
        pass

    @property
    @abstractmethod
    def ub(self) -> Optional[float]:
        """The upper bound of the interval or None for no upper bound"""
        pass

    @property
    @abstractmethod
    def strict_lb(self) -> bool:
        """Whether the lower bound is strict or not."""
        pass

    @property
    @abstractmethod
    def strict_ub(self) -> bool:
        """Whether the upper bound is strict or not."""
        pass

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


class CategoricalValue(Value, metaclass=ABCMeta):
    """Interface for a Value object with custom data type and categorical domain."""

    @property
    @abstractmethod
    def categories(self) -> list:
        """The discrete set of possible values defining the instance domain."""
        pass

    @property
    def domain(self) -> str:
        return str(self.categories)

    def in_domain(self, v) -> bool:
        return v in self.categories
