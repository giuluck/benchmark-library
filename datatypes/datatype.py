from abc import ABCMeta, abstractmethod
from typing import Optional


class DataType(metaclass=ABCMeta):
    """Interface for any Data Type object."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the instance, which must follow the python requirements about variables' names."""
        pass

    @property
    @abstractmethod
    def description(self) -> Optional[str]:
        """An optional textual description of the instance; default: None."""
        pass

    def __repr__(self) -> str:
        name = f"{self.__class__.__name__}({self.name})"
        return name if self.description is None else f"{name} -> {self.description}"
