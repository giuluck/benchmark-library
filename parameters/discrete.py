from typing import Optional, List

from parameters.parameter import Parameter
from utils import T


class DiscreteParameter(Parameter[T]):
    """A parameter with custom data type and discrete domain."""

    def __init__(self,
                 name: str,
                 default: T,
                 categories: List[T],
                 dtype: Optional[type] = None,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the parameter, which must follow the python requirements about variables' names.

        :param default:
            The default value for the parameter.

        :param categories:
            The discrete set of possible values defining the domain of the parameter.

        :param dtype:
            The type of the parameter, or None to infer it from the default value; default: None.
            In case generic types are used in class definition, the dtype should match them.

        :param description:
            An optional textual description of the parameter; default: None.
        """
        self.classes: List[T] = categories
        super(DiscreteParameter, self).__init__(default=default, name=name, dtype=dtype, description=description)

    def in_domain(self, value: T) -> bool:
        return value in self.classes

    def domain(self) -> Optional[str]:
        return str(self.classes)


class BinaryParameter(DiscreteParameter[int]):
    """A binary parameter which can only assume value 0 or 1."""

    def __init__(self, name: str, default: int, description: Optional[str] = None):
        """
        :param name:
            The name of the parameter, which must follow the python requirements about variables' names.

        :param default:
            The default value for the parameter.

        :param description:
            An optional textual description of the parameter; default: None.
        """
        super(BinaryParameter, self).__init__(
            default=default,
            name=name,
            dtype=int,
            description=description,
            categories=[0, 1]
        )
