from typing import Optional, Generic, Callable

from utils import T, to_string, none_string


class Parameter(Generic[T]):
    """Generic parameter for a benchmark."""

    def __init__(self, default: T, name: str, dtype: Optional[type] = None, description: Optional[str] = None):
        """
        :param name:
            The name of the parameter, which must follow the python requirements about variables' names.

        :param default:
            The default value for the parameter.

        :param dtype:
            The type of the parameter, or None to infer it from the default value; default: None.
            In case generic types are used in class definition, the dtype should match them.

        :param description:
            An optional textual description of the parameter; default: None.
        """
        self.name: str = name
        self.default: T = default
        self.dtype: type = dtype or type(default)
        self.description: Optional[str] = description
        self.validate(value=default)

    def __repr__(self) -> str:
        desc = f"{self.name} (type: {to_string(self.dtype)}, domain: {self.domain() or 'all'})"
        desc += none_string(self.description, prefix=": ")
        desc += f"; default = {to_string(self.default)}"
        return desc

    def in_domain(self, value: T) -> bool:
        """Checks if the given value is in the domain of the parameter.

        :param value:
            A possible value for the parameter.

        :return:
            A boolean value which indicates whether the assigned value falls or not into the domain of the parameter.
        """
        raise

    def domain(self) -> Optional[str]:
        """Describes the domain of the parameter.

        :return:
            Either a textual description of the domain in mathematical notation can be returned, or None if no domain.
        """
        raise NotImplementedError("Please implement abstract method ")

    def validate(self, value: T):
        """Validates a value assigned to the parameter.

        :param value:
            The value which is being assigned to the parameter.

        :raise `AssertionError`:
            If the value does not fall into the parameter's domain.
        """
        ins = isinstance(value, self.dtype)
        cst = self.in_domain(value)
        dom = self.domain() or "the constraint"
        assert ins, f"'{self.name}' should be of type {to_string(self.dtype)}, got type: {to_string(type(value))}"
        assert cst, f"'{self.name}' domain is {dom}, got value: {to_string(value)}"


class CustomParameter(Parameter[T]):
    """A parameter with custom domain."""

    def __init__(self,
                 name: str,
                 default: T,
                 in_domain: Optional[Callable] = None,
                 domain: Optional[str] = None,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the parameter, which must follow the python requirements about variables' names.

        :param default:
            The default value for the parameter.

        :param in_domain:
            A function f(value) -> bool which defines the domain of the parameter or None for no bounds; default: None.

        :param domain:
            An optional textual description of the domain; default: None.

        :param dtype:
            The type of the parameter, or None to infer it from the default value; default: None.
            In case generic types are used in class definition, the dtype should match them.

        :param description:
            An optional textual description of the parameter; default: None.
        """
        self._domain: Optional[str] = domain
        self._in_domain: Callable = in_domain or (lambda value: True)
        super(CustomParameter, self).__init__(default=default, name=name, dtype=dtype, description=description)

    def in_domain(self, value: T) -> bool:
        return self._in_domain(value)

    def domain(self) -> Optional[str]:
        return self._domain
