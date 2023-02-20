from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Callable

from datatypes.datatype import DataType


@dataclass(repr=False, frozen=True)
class Constraint(DataType, ABC):
    """Abstract class for global constraint on the input variables and parameters."""

    @abstractmethod
    def is_satisfied(self, v: Dict[str, Any], p: Dict[str, Any]) -> bool:
        """Checks if the global constraint is satisfied.

        :param v:
            The dictionary of benchmark variables indexed by name.

        :param p:
            The dictionary of benchmark parameters indexed by name.

        :return:
            Whether or not the global constraint is satisfied by the given assignment of variables and parameters.
        """
        pass


@dataclass(repr=False, frozen=True)
class CustomConstraint(Constraint):
    # default value is necessary to override abstract method from base class
    is_satisfied: Callable[[Dict[str, Any], Dict[str, Any]], bool] = field(default=lambda v, p: True, kw_only=True)
    """A function f(v, p) -> bool which checks if the global constraint is satisfied; default: f(v, p) -> True."""
