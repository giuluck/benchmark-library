from typing import Optional, Any

import numpy as np

from datatypes.parameters.parameter import Parameter
from datatypes.values import CategoricalValue


class CategoricalParameter(Parameter, CategoricalValue):
    """A parameter with custom data type and categorical domain."""

    def __init__(self,
                 name: str,
                 default: Any,
                 categories: list,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The list of value assignments that were taken by the variable during subsequent evaluation calls.

        :param categories:
            The discrete set of possible values defining the instance domain.

        :param dtype:
            The instance type, or None to infer it from the categories; default: None.
            If different data types are present in the categories, the 'object' data type is used instead.

        :param description:
            An optional textual description of the instance; default: None.
        """
        # handle dtype inference
        if dtype is None:
            dtype = type(categories[0])
            dtype = dtype if np.all([isinstance(c, dtype) for c in categories]) else object

        self._dtype = dtype
        self._categories: list = categories

        super(CategoricalParameter, self).__init__(name=name, default=default, description=description)

    @property
    def categories(self) -> list:
        return self._categories

    @property
    def dtype(self) -> type:
        return self._dtype


class BinaryParameter(CategoricalParameter):
    """A parameter which can only assume value 0 or 1."""

    def __init__(self, name: str, default: Any, description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The list of value assignments that were taken by the variable during subsequent evaluation calls.

        :param description:
            An optional textual description of the instance; default: None.
        """
        super(BinaryParameter, self).__init__(name=name,
                                              default=default,
                                              categories=[0, 1],
                                              dtype=int,
                                              description=description)
