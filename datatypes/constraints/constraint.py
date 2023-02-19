from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Any, Callable

from datatypes.datatype import DataType


class Constraint(DataType, metaclass=ABCMeta):
    def __init__(self, name: str, description: Optional[str]):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._name: str = name
        self._description: Optional[str] = description

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

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description


class CustomConstraint(Constraint):
    def __init__(self,
                 name: str,
                 is_satisfied: Callable[[Dict[str, Any], Dict[str, Any]], bool],
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param is_satisfied:
            A function f(v, p) -> bool which checks if the global constraint is satisfied.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._is_satisfied: Callable[[Dict[str, Any], Dict[str, Any]], bool] = is_satisfied
        super(CustomConstraint, self).__init__(name=name, description=description)

    def is_satisfied(self, var: Dict[str, Any], par: Dict[str, Any]) -> bool:
        return self._is_satisfied(var, par)
