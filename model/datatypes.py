from dataclasses import dataclass, field
from typing import Dict, Any
from typing import Optional, Callable

from model.utils import stringify


@dataclass(frozen=True, repr=False)
class Sample:
    """Basic dataclass representing a benchmark sample, which contains inputs and output values."""

    inputs: Dict[str, Any] = field(kw_only=True)
    """The dictionary of input variables and parameters."""

    output: Any = field(kw_only=True)
    """The output object, which can be of any type."""

    def __getitem__(self, key):
        match key:
            case 0 | 'inputs':
                return self.inputs
            case 1 | 'output':
                return self.output
            case _:
                raise IndexError(f"Sample instances admit keys [0, 1, 'inputs', 'output'], got {stringify(key)}'")

    def __repr__(self):
        return f"Sample(inputs={stringify(self.inputs)}, output={stringify(self.output)})"


@dataclass(frozen=True, repr=False)
class DataType:
    """Base class for any benchmark object."""

    name: str = field(kw_only=True)
    """The name of the instance, which must follow the python requirements about variables' names."""

    description: Optional[str] = field(kw_only=True)
    """An optional textual description of the instance."""

    def __post_init__(self):
        assert self.name.isidentifier(), f"Invalid name '{self.name}'"

    def __repr__(self) -> str:
        name = f"{self.__class__.__name__}({self.name})"
        return name if self.description is None else f"{name} -> {self.description}"


@dataclass(frozen=True, repr=False)
class Input(DataType):
    """A benchmark input, i.e., either a variable or a parameter."""

    dtype: type = field(kw_only=True)
    """The instance type."""

    domain: str = field(kw_only=True)
    """A textual description of the domain."""

    check_domain: Callable[[Any], bool] = field(kw_only=True)
    """Checks if a given value is in the variable domain."""

    def check_dtype(self, value: Any) -> bool:
        """Checks if a given value has the correct type."""
        return isinstance(value, (int, float)) if self.dtype is float else isinstance(value, self.dtype)


@dataclass(frozen=True, repr=False)
class Variable(Input):
    """A benchmark variable."""


@dataclass(frozen=True, repr=False)
class Parameter(Input):
    """A benchmark parameter."""

    def __post_init__(self):
        default = stringify(self.default)
        assert self.check_dtype(self.default), f"Parameter '{self.name}' should be of type {self.dtype.__name__}, got" \
                                               f" value {default} of type {type(self.default).__name__}"
        assert self.check_domain(self.default), f"Parameter '{self.name}' domain is {self.domain}, got value {default}"

    default: Any = field(kw_only=True)
    """The default value of the parameter."""


@dataclass(frozen=True, repr=False)
class Constraint(DataType):
    """A global constraint on the input variables and parameters."""

    check: Callable[..., bool] = field(kw_only=True)
    """Checks if the global constraint is satisfied given the input values of variables and parameters."""


@dataclass(frozen=True, repr=False)
class Metric(DataType):
    """A benchmark metric."""

    function: Callable[[Sample], float] = field()

    def __call__(self, sample: Sample) -> float:
        return self.function(sample)
