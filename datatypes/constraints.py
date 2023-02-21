import inspect
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any, Callable

from datatypes.datatype import DataType


@dataclass(repr=False, frozen=True)
class Constraint(DataType, ABC):
    """Abstract class for global constraint on the input variables and parameters."""

    @abstractmethod
    def is_satisfied(self, **inputs: Any) -> bool:
        """Checks if the global constraint is satisfied.

        :param inputs:
            The input values, containing both variables and parameters instances, indexed by name.

        :return:
            Whether or not the global constraint is satisfied by the given assignment of variables and parameters.
        """
        pass


# noinspection PyDataclass
@dataclass(repr=False, frozen=True)
class GenericConstraint(Constraint):
    """A constraint with custom function."""

    satisfied_fn: Callable = field(kw_only=True)
    """A function f(...) -> bool which checks if the global constraint is satisfied. The input parameters must match
    the names of either the benchmark variables or the benchmark parameters."""

    def is_satisfied(self, **inputs) -> bool:
        params = inspect.signature(self.satisfied_fn).parameters
        inputs = {inp: val for inp, val in inputs.items() if inp in params}
        return self.satisfied_fn(**inputs)
