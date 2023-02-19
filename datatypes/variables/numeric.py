from typing import Optional

from datatypes.values import NumericValue
from datatypes.variables.variable import Variable


class NumericVariable(Variable, NumericValue):
    """A variable with numeric type and interval-based domain."""

    def __init__(self,
                 name: str,
                 integer: bool = False,
                 internal: bool = True,
                 lb: Optional[float] = None,
                 ub: Optional[float] = None,
                 strict_lb: bool = False,
                 strict_ub: bool = False,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param integer:
            Whether the value is expected to be an integer or a real number.

        :param internal:
            Whether the domain is internal to the bounds (i.e., [lb, ub]) or external (i.e., [-inf, lb] U [ub, +inf]).

        :param lb:
            The lower bound of the interval or None for no lower bound; default: None.

        :param ub:
            The upper bound of the interval or None for no upper bound; default: None.

        :param strict_lb:
            Whether the lower bound is strict or not; default = False.

        :param strict_ub:
            Whether the upper bound is strict or not; default = False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert lb is None or ub is None or ub >= lb, f"'ub' must be greater or equal to 'lb', got {ub} < {lb}"
        self._integer: bool = integer
        self._internal: bool = internal
        self._lb: Optional[float] = lb
        self._ub: Optional[float] = ub
        self._strict_lb: bool = strict_lb
        self._strict_ub: bool = strict_ub
        super(NumericVariable, self).__init__(name=name, description=description)

    @property
    def integer(self) -> bool:
        return self._integer

    @property
    def internal(self) -> bool:
        return self._internal

    @property
    def lb(self) -> Optional[float]:
        return self._lb

    @property
    def ub(self) -> Optional[float]:
        return self._ub

    @property
    def strict_lb(self) -> bool:
        return self._strict_lb

    @property
    def strict_ub(self) -> bool:
        return self._strict_ub


class PositiveVariable(NumericVariable):
    """A variable with numeric type which can assume positive values only."""

    def __init__(self,
                 name: str,
                 integer: bool = False,
                 strict: bool = False,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param integer:
            Whether the value is expected to be an integer or a real number.

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        super(PositiveVariable, self).__init__(
            name=name,
            integer=integer,
            internal=True,
            lb=0.0,
            ub=None,
            strict_lb=strict,
            description=description
        )


class NegativeVariable(NumericVariable):
    """A variable with numeric type which can assume negative values only."""

    def __init__(self,
                 name: str,
                 integer: bool = False,
                 strict: bool = False,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param integer:
            Whether the value is expected to be an integer or a real number.

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        super(NegativeVariable, self).__init__(
            name=name,
            integer=integer,
            internal=True,
            lb=None,
            ub=0.0,
            strict_ub=strict,
            description=description
        )


class ThresholdVariable(NumericVariable):
    """A variable with numeric type which must be smaller than a certain threshold in absolute value."""

    def __init__(self,
                 name: str,
                 threshold: float,
                 integer: bool = False,
                 strict: bool = False,
                 description: Optional[str] = None):
        """
        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param threshold:
            A positive value representing the threshold.

        :param integer:
            Whether the value is expected to be an integer or a real number.

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert threshold > 0, f"threshold should be a positive value, got {threshold}"
        super(ThresholdVariable, self).__init__(
            name=name,
            integer=integer,
            internal=True,
            lb=-threshold,
            ub=threshold,
            strict_lb=strict,
            strict_ub=strict,
            description=description
        )
