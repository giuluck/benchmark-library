from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from utils.strings import stringify


@dataclass(frozen=True)
class Sample:
    """Basic dataclass representing a benchmark sample, which contains inputs and output values."""

    inputs: Dict[str, Any]
    """The dictionary of input variables and parameters."""

    output: Any
    """The output object, which can be of any type."""

    def __getitem__(self, key):
        match key:
            case 0 | 'inputs':
                return self.inputs
            case 1 | 'output':
                return self.output
            case _:
                raise IndexError(f"Sample instances admit keys [0, 1, 'inputs', 'output'], got {stringify(key)}'")


@dataclass(repr=False, frozen=True)
class DataType:
    """Base class for any Data Type object."""

    name: str = field()
    """The name of the instance, which must follow the python requirements about variables' names."""

    description: Optional[str] = field(default=None, kw_only=True)
    """An optional textual description of the instance; default: None."""

    def __post_init__(self):
        assert self.name.isidentifier(), f"Invalid instance name '{self.name}'"

    def __repr__(self) -> str:
        name = f"{self.__class__.__name__}({self.name})"
        return name if self.description is None else f"{name} -> {self.description}"
