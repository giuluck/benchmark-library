from dataclasses import dataclass, field
from typing import Optional


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
