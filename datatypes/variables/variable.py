from abc import ABCMeta
from typing import Optional, Any, Callable

from datatypes.values import Value


class Variable(Value, metaclass=ABCMeta):
    """Abstract template for a Variable."""

    def __init__(self, name: str, description: Optional[str]):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._name: str = name
        self._description: Optional[str] = description
        self._assignments: list = []

    @property
    def assignments(self) -> list:
        """The list of value assignments that were taken by the variable during subsequent evaluation calls."""
        return self._assignments

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description


class CustomVariable(Variable):
    """A variable with custom domain and dtype."""

    def __init__(self,
                 name: str,
                 dtype: type = object,
                 domain: str = 'all',
                 in_domain: Callable[[Any], bool] = lambda v: True,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param dtype:
            The instance type; default: object.

        :param domain:
            A textual description of the domain; default: 'all'.

        :param in_domain:
            A function f(x) -> bool which defines whether the variable domain; default: f(x) -> True.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._dtype: type = dtype
        self._domain: str = domain
        self._in_domain: Callable[[Any], bool] = in_domain
        super(CustomVariable, self).__init__(name=name, description=description)

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def domain(self) -> str:
        return self._domain

    def in_domain(self, v) -> bool:
        return self._in_domain(v)
