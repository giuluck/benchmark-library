from abc import ABCMeta
from typing import Optional, Any, Callable

from datatypes.values import Value
from utils.strings import stringify


class Parameter(Value, metaclass=ABCMeta):
    """Abstract template for a Variable."""

    def __init__(self, name: str, default: Any, description: Optional[str]):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The list of value assignments that were taken by the variable during subsequent evaluation calls.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._name: str = name
        self._default: Any = default
        self._description: Optional[str] = description
        assert isinstance(default, self.dtype), f"'{name}' should be of type {stringify(self.dtype)}, got default " \
                                                f"value {stringify(default)} of type {stringify(type(default))}"
        assert self.in_domain(default), f"'{name}' domain is {self.domain}, got default value {default}"

    @property
    def default(self) -> Any:
        """The list of value assignments that were taken by the variable during subsequent evaluation calls."""
        return self._default

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    def __repr__(self):
        representation = super(Parameter, self).__repr__()
        return f"{representation}; default: {self.default}"


class CustomParameter(Parameter):
    """A parameter with custom domain and dtype."""

    def __init__(self,
                 name: str,
                 default: Any,
                 dtype: type = object,
                 domain: str = 'all',
                 in_domain: Callable[[Any], bool] = lambda v: True,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The list of value assignments that were taken by the variable during subsequent evaluation calls.

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
        super(CustomParameter, self).__init__(name=name, default=default, description=description)

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def domain(self) -> str:
        return self._domain

    def in_domain(self, v) -> bool:
        return self.in_domain(v)
