from typing import Optional

from parameters.parameter import Parameter
from utils import N


class IntervalParameter(Parameter[N]):
    """A parameter with numeric type and interval-based domain."""

    def __init__(self,
                 name: str,
                 default: N,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None,
                 lb: Optional[N] = None,
                 ub: Optional[N] = None,
                 strict_lb: bool = False,
                 strict_ub: bool = False,
                 internal: bool = True):
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

        :param lb:
            The lower bound of the interval or None for no lower bound; default: None.

        :param ub:
            The upper bound of the interval or None for no upper bound; default: None.

        :param strict_lb:
            Whether the lower bound is strict or not; default = False.

        :param strict_ub:
            Whether the upper bound is strict or not; default = False.

        :param internal:
            Whether the domain is within ([lb, ub]) or outside the interval (]-inf, lb] or [ub, +inf[); default = True.
        """
        self.lb: Optional[N] = lb
        self.ub: Optional[N] = ub
        self.strict_lb: bool = strict_lb
        self.strict_ub: bool = strict_ub
        self.internal: bool = internal
        super(IntervalParameter, self).__init__(default=default, name=name, dtype=dtype, description=description)

    def in_domain(self, value: N) -> bool:
        # handle all the scenarios (half-bounded vs. double bounded; strict vs. non-strict; internal vs. external)
        match (self.lb, self.ub, self.strict_lb, self.strict_ub, self.internal):
            case (None, None, _, _, _):
                # no bounds
                return True
            case (lb, None, True, _, True):
                # internal strict lower bound only
                return value > lb
            case (lb, None, False, _, True):
                # internal non-strict lower bound only
                return value >= lb
            case (None, ub, _, True, True):
                # internal strict upper bound only
                return value < ub
            case (None, ub, _, False, True):
                # internal non-strict upper bound only
                return value <= ub
            case (lb, ub, True, True, True):
                # internal strict lb and ub
                return lb < value < ub
            case (lb, ub, False, True, True):
                # internal non-strict lb strict ub
                return lb <= value < ub
            case (lb, ub, True, False, True):
                # internal strict lb non-strict ub
                return lb < value <= ub
            case (lb, ub, False, False, True):
                # internal non-strict lb and ub
                return lb <= value <= ub
            case (lb, None, True, _, False):
                # external strict lower bound only
                return value < lb
            case (lb, None, False, _, False):
                # external non-strict lower bound only
                return value <= lb
            case (None, ub, _, True, False):
                # external strict upper bound only
                return value > ub
            case (None, ub, _, False, False):
                # external non-strict upper bound only
                return value >= ub
            case (lb, ub, True, True, False):
                # external strict lb and ub
                return value < lb or value > ub
            case (lb, ub, False, True, False):
                # external non-strict lb strict ub
                return value <= lb or value > ub
            case (lb, ub, True, False, False):
                # external strict lb non-strict ub
                return value < lb or value >= ub
            case (lb, ub, False, False, False):
                # external non-strict lb and ub
                return value <= lb or value >= ub
            case _:
                raise RuntimeError("Something went wrong during constraint checks")

    def domain(self) -> Optional[str]:
        # handle all the scenarios (half-bounded vs. double bounded; strict vs. non-strict; internal vs. external)
        match (self.lb, self.ub, self.strict_lb, self.strict_ub, self.internal):
            case (None, None, _, _, _):
                # no bounds
                return None
            case (lb, None, True, _, True):
                # internal strict lower bound only
                return f"x > {lb}"
            case (lb, None, False, _, True):
                # internal non-strict lower bound only
                return f"x >= {lb}"
            case (None, ub, _, True, True):
                # internal strict upper bound only
                return f"x < {ub}"
            case (None, ub, _, False, True):
                # internal non-strict upper bound only
                return f"x <= {ub}"
            case (lb, ub, True, True, True):
                # internal strict lb and ub
                return f"]{lb}, {ub}["
            case (lb, ub, False, True, True):
                # internal non-strict lb strict ub
                return f"[{lb}, {ub}["
            case (lb, ub, True, False, True):
                # internal strict lb non-strict ub
                return f"]{lb}, {ub}]"
            case (lb, ub, False, False, True):
                # internal non-strict lb and ub
                return f"[{lb}, {ub}]"
            case (lb, None, True, _, False):
                # external strict lower bound only
                return f"out of [{lb}, +inf["
            case (lb, None, False, _, False):
                # external non-strict lower bound only
                return f"out of ]{lb}, +inf["
            case (None, ub, _, True, False):
                # external strict upper bound only
                return f"out of ]-inf, {ub}]"
            case (None, ub, _, False, False):
                # external non-strict upper bound only
                return f"out of ]-inf, {ub}["
            case (lb, ub, True, True, False):
                # external strict lb and ub
                return f"out of [{lb}, {ub}]"
            case (lb, ub, False, True, False):
                # external non-strict lb strict ub
                return f"out of ]{lb}, {ub}]"
            case (lb, ub, True, False, False):
                # external strict lb non-strict ub
                return f"out of [{lb}, {ub}["
            case (lb, ub, False, False, False):
                # external non-strict lb and ub
                return f"out of ]{lb}, {ub}["
            case _:
                raise RuntimeError("Something went wrong during constraint checks")


class PositiveParameter(IntervalParameter[N]):
    """A parameter with numeric type which can assume positive values only."""

    def __init__(self,
                 name: str,
                 default: N,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None,
                 strict: bool = False):
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

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.
        """
        super(PositiveParameter, self).__init__(
            default=default,
            name=name,
            dtype=dtype,
            description=description,
            lb=0,
            ub=None,
            strict_lb=strict,
            internal=True
        )


class NegativeParameter(IntervalParameter[N]):
    """A parameter with numeric type which can assume negative values only."""

    def __init__(self,
                 name: str,
                 default: N,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None,
                 strict: bool = False):
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

        :param strict:
            Whether the value must be strictly negative or it can have value zero; default = False.
        """
        super(NegativeParameter, self).__init__(
            default=default,
            name=name,
            dtype=dtype,
            description=description,
            lb=None,
            ub=0,
            strict_ub=strict,
            internal=True
        )


class ThresholdParameter(IntervalParameter[N]):
    """A parameter with numeric type which must be smaller than a certain threshold in absolute value."""

    def __init__(self,
                 name: str,
                 default: N,
                 threshold: N,
                 dtype: Optional[type] = None,
                 description: Optional[str] = None,
                 strict: bool = False):
        """
        :param name:
            The name of the parameter, which must follow the python requirements about variables' names.

        :param default:
            The default value for the parameter.

        :parameter threshold:
            A positive value representing the threshold.

        :param dtype:
            The type of the parameter, or None to infer it from the default value; default: None.
            In case generic types are used in class definition, the dtype should match them.

        :param description:
            An optional textual description of the parameter; default: None.

        :param strict:
            Whether the interval bounds are strict or not; default = False.
        """
        assert threshold > 0, f"threshold should be a positive value, got {threshold}"
        super(ThresholdParameter, self).__init__(
            default=default,
            name=name,
            dtype=dtype,
            description=description,
            lb=-threshold,
            ub=threshold,
            strict_lb=strict,
            strict_ub=strict,
            internal=True
        )
