from typing import List, Callable, Any, Optional, Tuple, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss, precision_score, \
    recall_score, f1_score, accuracy_score, roc_auc_score

from model.datatypes import Variable, Parameter, Constraint, Metric, Sample, Input, DataType
from model.utils import stringify


class Structure:
    """Data class representing the internal structure of a benchmark, namely its datatypes and other properties."""

    # ----------------------------------------- STRUCTURE UTILITIES START ----------------------------------------------

    _metric_aliases: Dict[str, Callable] = dict(
        mae=lambda reference, value: mean_absolute_error(np.atleast_1d(reference), np.atleast_1d(value)),
        mse=lambda reference, value: mean_squared_error(np.atleast_1d(reference), np.atleast_1d(value)),
        r2=lambda reference, value: r2_score(np.atleast_1d(reference), np.atleast_1d(value)),
        crossentropy=lambda reference, value: log_loss(np.atleast_1d(reference), np.atleast_1d(value)),
        precision=lambda reference, value: precision_score(np.atleast_1d(reference), np.atleast_1d(value)),
        recall=lambda reference, value: recall_score(np.atleast_1d(reference), np.atleast_1d(value)),
        f1=lambda reference, value: f1_score(np.atleast_1d(reference), np.atleast_1d(value)),
        accuracy=lambda reference, value: accuracy_score(np.atleast_1d(reference), np.atleast_1d(value)),
        auc=lambda reference, value: roc_auc_score(np.atleast_1d(reference), np.atleast_1d(value))
    )

    @staticmethod
    def _get_numeric_domain(lb: Optional[float],
                            ub: Optional[float],
                            strict_lb: bool,
                            strict_ub: bool,
                            internal: bool) -> Tuple[Optional[str], Callable[[float], bool]]:
        # handle all the scenarios (half-bounded vs. double bounded; strict vs. non-strict; internal vs. external)
        match (lb, ub, strict_lb, strict_ub, internal):
            case (None, None, _, _, _):
                # no bounds
                return None, lambda value: True
            case (lb, None, True, _, True):
                # internal strict lower bound only
                return f"]{lb}, +inf[", lambda value: value > lb
            case (lb, None, False, _, True):
                # internal non-strict lower bound only
                return f"[{lb}, +inf[", lambda value: value >= lb
            case (None, ub, _, True, True):
                # internal strict upper bound only
                return f"]-inf, {ub}[", lambda value: value < ub
            case (None, ub, _, False, True):
                # internal non-strict upper bound only
                return f"]-inf, {ub}]", lambda value: value <= ub
            case (lb, ub, True, True, True):
                # internal strict lb and ub
                return f"]{lb}, {ub}[", lambda value: lb < value < ub
            case (lb, ub, False, True, True):
                # internal non-strict lb strict ub
                return f"[{lb}, {ub}[", lambda value: lb <= value < ub
            case (lb, ub, True, False, True):
                # internal strict lb non-strict ub
                return f"]{lb}, {ub}]", lambda value: lb < value <= ub
            case (lb, ub, False, False, True):
                # internal non-strict lb and ub
                return f"[{lb}, {ub}]", lambda value: lb <= value <= ub
            case (lb, None, True, _, False):
                # external strict lower bound only
                return f"out of [{lb}, +inf[", lambda value: value < lb
            case (lb, None, False, _, False):
                # external non-strict lower bound only
                return f"out of ]{lb}, +inf[", lambda value: value <= lb
            case (None, ub, _, True, False):
                # external strict upper bound only
                return f"out of ]-inf, {ub}]", lambda value: value > ub
            case (None, ub, _, False, False):
                # external non-strict upper bound only
                return f"out of ]-inf, {ub}[", lambda value: value >= ub
            case (lb, ub, True, True, False):
                # external strict lb and ub
                return f"out of [{lb}, {ub}]", lambda value: value < lb or value > ub
            case (lb, ub, False, True, False):
                # external non-strict lb strict ub
                return f"out of ]{lb}, {ub}]", lambda value: value <= lb or value > ub
            case (lb, ub, True, False, False):
                # external strict lb non-strict ub
                return f"out of [{lb}, {ub}[", lambda value: value < lb or value >= ub
            case (lb, ub, False, False, False):
                # external non-strict lb and ub
                return f"out of ]{lb}, {ub}[", lambda value: value <= lb or value >= ub
            case _:
                raise RuntimeError("Something went wrong during constraint checks")

    def _add_var(self, name: str, dtype: type, check: Callable, domain: str, description: Optional[str]):
        assert name not in self._variables, f"Name '{name}' is already assigned to another variable"
        assert name not in self._parameters, f"Name '{name}' is already assigned to another parameter"
        self._variables[name] = Variable(
            name=name,
            description=description,
            dtype=dtype,
            domain=domain,
            check_domain=check
        )

    def _add_par(self, name: str, default: Any, dtype: type, check: Callable, domain: str, description: Optional[str]):
        assert name not in self._parameters, f"Name '{name}' is already assigned to another parameter"
        assert name not in self._variables, f"Name '{name}' is already assigned to another variable"
        self._parameters[name] = Parameter(
            name=name,
            default=default,
            description=description,
            dtype=dtype,
            domain=domain,
            check_domain=check
        )

    def _add_cst(self, name: str, check: Callable, description: Optional[str]):
        assert name not in self._constraints, f"Name '{name}' is already assigned to another constraint"
        self._constraints[name] = Constraint(name=name, description=description, check=check)

    def _add_mtr(self, name: str, function: Callable, description: Optional[str]):
        assert name not in self._metrics, f"Name '{name}' is already assigned to another metric"
        self._metrics[name] = Metric(name=name, description=description, function=function)

    # ------------------------------------------ STRUCTURE UTILITIES END -----------------------------------------------
    # ----------------------------------------- STRUCTURE PROPERTIES START ---------------------------------------------

    def __init__(self, alias: str, description: str):
        """
        :param alias:
            The benchmark alias.

        :param description:
            The benchmark description.
        """
        self.alias: str = alias
        """The benchmark alias."""

        self.description: str = description
        """The benchmark description."""

        self._variables: Dict[str, Variable] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._constraints: Dict[str, Constraint] = {}
        self._metrics: Dict[str, Metric] = {}

    @property
    def variables(self) -> List[Variable]:
        """The list of benchmark variables."""
        return list(self._variables.values())

    @property
    def parameters(self) -> List[Parameter]:
        """The list of benchmark variables."""
        return list(self._parameters.values())

    @property
    def constraints(self) -> List[Constraint]:
        """The list of benchmark constraints."""
        return list(self._constraints.values())

    @property
    def metrics(self) -> List[Metric]:
        """The list of benchmark metrics."""
        return list(self._metrics.values())

    @property
    def inputs(self) -> List[Input]:
        """The list of benchmark inputs, i.e., its variables and parameters."""
        return [*self._variables.values(), *self._parameters.values()]

    @property
    def datatypes(self) -> List[DataType]:
        """The list of benchmark datatypes, i.e., its variables, parameters, constraints, and metrics."""
        return [
            *self._variables.values(),
            *self._parameters.values(),
            *self._constraints.values(),
            *self._metrics.values()
        ]

    # ------------------------------------------ STRUCTURE PROPERTIES END ----------------------------------------------
    # ------------------------------------------- VARIABLES METHODS START ----------------------------------------------

    def add_custom_variable(self,
                            name: str,
                            dtype: type = object,
                            check: Callable[[Any], bool] = lambda v: True,
                            domain: str = 'all',
                            description: Optional[str] = None):
        """Adds a new variable with custom domain and dtype.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param dtype:
            The instance type; default: object.

        :param domain:
            A textual description of the domain; default: 'all'.

        :param check:
            A function f(x) -> bool which defines whether the variable domain; default: f(x) -> True.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_numeric_variable(self,
                             name: str,
                             integer: bool = False,
                             internal: bool = True,
                             lb: Optional[float] = None,
                             ub: Optional[float] = None,
                             strict_lb: bool = False,
                             strict_ub: bool = False,
                             description: Optional[str] = None):
        """Adds a new variable numeric type and interval-based domain.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param internal:
            Whether the domain is internal (i.e., [lb, ub]) or external (i.e., ]-inf, lb] U [ub, +inf[); default: True.

        :param lb:
            The lower bound of the interval or None for no lower bound; default: None.

        :param ub:
            The upper bound of the interval or None for no upper bound; default: None.

        :param strict_lb:
            Whether the lower bound is strict or not; default: False.

        :param strict_ub:
            Whether the upper bound is strict or not; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert lb is None or ub is None or ub >= lb, f"'ub' must be greater or equal to 'lb', got {ub} < {lb}"
        domain, check = self._get_numeric_domain(lb, ub, strict_lb, strict_ub, internal)
        dtype = int if integer else float
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_positive_variable(self,
                              name: str,
                              strict: bool = False,
                              integer: bool = False,
                              description: Optional[str] = None):
        """Adds a new variable with numeric type which can assume positive values only.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if strict:
            domain, check = "]0, +inf[", lambda value: value > 0
        else:
            domain, check = "[0, +inf[", lambda value: value >= 0
        dtype = int if integer else float
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_negative_variable(self,
                              name: str,
                              strict: bool = False,
                              integer: bool = False,
                              description: Optional[str] = None):
        """Adds a new variable with numeric type which can assume negative values only.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param strict:
            Whether the value must be strictly negative or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if strict:
            domain, check = "]-inf, 0[", lambda value: value < 0
        else:
            domain, check = "]-inf, 0]", lambda value: value <= 0
        dtype = int if integer else float
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_threshold_variable(self,
                               name: str,
                               threshold: float,
                               strict: bool = False,
                               integer: bool = False,
                               description: Optional[str] = None):
        """Adds a new variable with numeric type which must be smaller than a certain threshold in absolute value.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param threshold:
            A positive value representing the threshold.

        :param strict:
            Whether the value must be strictly negative or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert threshold > 0, f"threshold should be a positive value, got {threshold}"
        if strict:
            domain, check = f"]-{threshold}, {threshold}[", lambda value: -threshold < value < threshold
        else:
            domain, check = f"[-{threshold}, {threshold}]", lambda value: value <= 0
        dtype = int if integer else float
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_categorical_variable(self,
                                 name: str,
                                 categories: set | list | tuple,
                                 dtype: Optional[type] = None,
                                 description: Optional[str] = None):
        """Adds a new variable with custom data type and categorical domain.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param categories:
            The discrete set of possible values defining the instance domain.

        :param dtype:
            The expected instance type, or None to infer it from the categories; default: None.

        :param description:
            An optional textual description of the instance; default: None.
        """
        # if dtype is None, tries to infer it from the categories, i.e.:
        #   - if all the objects in the set have share the same type, then that type is used
        #   - otherwise, the 'object' type is used instead
        if dtype is None:
            dtypes = [type(cat) for cat in categories]
            dtype = dtypes[0] if all([dt is dtypes[0] for dt in dtypes]) else object
        domain = stringify(list(categories))
        check = lambda value: value in categories
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    def add_binary_variable(self, name: str, description: Optional[str]):
        """Adds a new variable which can only assume value 0 or 1.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param description:
            An optional textual description of the instance; default: None.
        """
        domain, check, dtype = "[0, 1]", lambda value: value in [0, 1], int
        self._add_var(name=name, dtype=dtype, check=check, domain=domain, description=description)

    # --------------------------------------------- VARIABLES METHODS END ----------------------------------------------
    # -------------------------------------------- PARAMETERS METHODS START --------------------------------------------

    def add_custom_parameter(self,
                             name: str,
                             default: Any,
                             dtype: type = object,
                             check: Callable[[Any], bool] = lambda v: True,
                             domain: str = 'all',
                             description: Optional[str] = None):
        """
        Adds a new parameter with custom domain and dtype.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param dtype:
            The instance type; default: object.

        :param domain:
            A textual description of the domain; default: 'all'.

        :param check:
            A function f(x) -> bool which defines whether the parameter domain; default: f(x) -> True.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_numeric_parameter(self,
                              name: str,
                              default: Any,
                              integer: bool = False,
                              internal: bool = True,
                              lb: Optional[float] = None,
                              ub: Optional[float] = None,
                              strict_lb: bool = False,
                              strict_ub: bool = False,
                              description: Optional[str] = None):
        """Adds a new parameter numeric type and interval-based domain.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param internal:
            Whether the domain is internal (i.e., [lb, ub]) or external (i.e., ]-inf, lb] U [ub, +inf[); default: True.

        :param lb:
            The lower bound of the interval or None for no lower bound; default: None.

        :param ub:
            The upper bound of the interval or None for no upper bound; default: None.

        :param strict_lb:
            Whether the lower bound is strict or not; default: False.

        :param strict_ub:
            Whether the upper bound is strict or not; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert lb is None or ub is None or ub >= lb, f"'ub' must be greater or equal to 'lb', got {ub} < {lb}"
        domain, check = self._get_numeric_domain(lb, ub, strict_lb, strict_ub, internal)
        dtype = int if integer else float
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_positive_parameter(self,
                               name: str,
                               default: Any,
                               strict: bool = False,
                               integer: bool = False,
                               description: Optional[str] = None):
        """Adds a new parameter with numeric type which can assume positive values only.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param strict:
            Whether the value must be strictly positive or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if strict:
            domain, check = "]0, +inf[", lambda value: value > 0
        else:
            domain, check = "[0, +inf[", lambda value: value >= 0
        dtype = int if integer else float
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_negative_parameter(self,
                               name: str,
                               default: Any,
                               strict: bool = False,
                               integer: bool = False,
                               description: Optional[str] = None):
        """Adds a new parameter with numeric type which can assume negative values only.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param strict:
            Whether the value must be strictly negative or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if strict:
            domain, check = "]-inf, 0[", lambda value: value < 0
        else:
            domain, check = "]-inf, 0]", lambda value: value <= 0
        dtype = int if integer else float
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_threshold_parameter(self,
                                name: str,
                                default: Any,
                                threshold: float,
                                strict: bool = False,
                                integer: bool = False,
                                description: Optional[str] = None):
        """Adds a new parameter with numeric type which must be smaller than a certain threshold in absolute value.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param threshold:
            A positive value representing the threshold.

        :param strict:
            Whether the value must be strictly negative or it can have value zero; default = False.

        :param integer:
            Whether the value is expected to be an integer or a real number; default: False.

        :param description:
            An optional textual description of the instance; default: None.
        """
        assert threshold > 0, f"threshold should be a positive value, got {threshold}"
        if strict:
            domain, check = f"]-{threshold}, {threshold}[", lambda value: -threshold < value < threshold
        else:
            domain, check = f"[-{threshold}, {threshold}]", lambda value: value <= 0
        dtype = int if integer else float
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_categorical_parameter(self,
                                  name: str,
                                  default: Any,
                                  categories: set | list | tuple,
                                  dtype: Optional[type] = None,
                                  description: Optional[str] = None):
        """Adds a new parameter with custom data type and categorical domain.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param categories:
            The discrete set of possible values defining the instance domain.

        :param dtype:
            The expected instance type, or None to infer it from the categories; default: None.

        :param description:
            An optional textual description of the instance; default: None.
        """
        # if dtype is None, tries to infer it from the categories, i.e.:
        #   - if all the objects in the set have share the same type, then that type is used
        #   - otherwise, the 'object' type is used instead
        if dtype is None:
            dtypes = [type(cat) for cat in categories]
            dtype = dtypes[0] if all([dt is dtypes[0] for dt in dtypes]) else object
        domain = stringify(list(categories))
        check = lambda value: value in categories
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    def add_binary_parameter(self, name: str, default: bool, description: Optional[str]):
        """Adds a new parameter which can only assume value 0 or 1.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param default:
            The default value of the parameter.

        :param description:
            An optional textual description of the instance; default: None.
        """
        domain, check, dtype = "[0, 1]", lambda value: value in [0, 1], int
        self._add_par(name=name, default=default, description=description, dtype=dtype, domain=domain, check=check)

    # -------------------------------------------- PARAMETERS METHODS END ----------------------------------------------
    # ------------------------------------------- CONSTRAINTS METHODS START --------------------------------------------

    def add_generic_constraint(self, name: str, check: Callable[..., bool], description: Optional[str] = None):
        """Adds a new parameter which can only assume value 0 or 1.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param check:
            The default value of the parameter.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._add_cst(name=name, check=check, description=description)

    # --------------------------------------------- CONSTRAINTS METHODS END --------------------------------------------
    # ---------------------------------------------- METRICS METHODS START ---------------------------------------------

    def add_sample_metric(self, name: str, function: Callable[[Sample], bool], description: Optional[str] = None):
        """A metric with custom function computed on a given sample.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param function:
            A function f(sample) -> float which computes the metric on a given sample <inputs, output>.

        :param description:
            An optional textual description of the instance; default: None.
        """
        self._add_mtr(name=name, function=function, description=description)

    def add_value_metric(self,
                         name: str,
                         function: Callable[[Any], bool],
                         value: str = 'output',
                         description: Optional[str] = None):
        """A metric with custom function computed on a single value of the given sample.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param function:
            A function f(value) -> float which computes the metric on the specified value of the given sample.

        :param value:
            The input value on which to compute the metric, or 'output' to compute it on the output; default: 'output'.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if value == 'output':
            func = lambda sample: function(sample.output)
        else:
            func = lambda sample: function(sample.inputs[value])
        self._add_mtr(name=name, function=func, description=description)

    def add_reference_metric(self,
                             name: str,
                             metric: str | Callable[[Any], bool],
                             reference: Any,
                             value: str = 'output',
                             description: Optional[str] = None):
        """A metric computed on a single value with respect to a reference value.

        :param name:
            The name of the instance, which must follow the python requirements about variables' names.

        :param metric:
            Either a metric alias or a function f(value) -> float.

        :param reference:
            The reference output value on which to compare the specified value of the given sample.

        :param value:
            The input value on which to compute the metric, or 'output' to compute it on the output; default: 'output'.

        :param description:
            An optional textual description of the instance; default: None.
        """
        if isinstance(metric, str):
            alias = metric
            metric = self._metric_aliases.get(metric)
            assert metric is not None, f"Unknown function alias '{alias}'"
        if value == 'output':
            func = lambda sample: metric(reference, sample.output)
        else:
            func = lambda sample: metric(reference, sample.inputs[value])
        self._add_mtr(name=name, function=func, description=description)
