import inspect
from inspect import Parameter
from typing import Any

from benchmarks.benchmark import Benchmark


class BenchmarkDecorator:
    """Decorator for a benchmark instance. It changes the signature of the '__init__' and 'query'  methods in order to
    include the benchmark-specific inputs (i.e., variables and parameters instances). Even though the decorator takes
    care of changing the input signature of the '__init__' and the 'query' methods, overriding them with the correct
    parameter names is helpful for autocompletion in many IDEs. Similarly, the decorator also takes care of the default
    values and annotations, but adding them will be beneficial for static type checking and signatures. Therefore, the
    decorator can be used to find the wanted trade-off between detailed static API and boilerplate/maintenance costs.
    """

    def __call__(self, target) -> Any:
        """Changes the signature of the '__init__' and 'query'  methods in order to include the benchmark inputs.

        :return:
            The target itself.
        """

        def _parameter(obj) -> Parameter:
            default = {'default': obj.default} if hasattr(obj, 'default') else {}
            return Parameter(name=obj.name, annotation=obj.dtype, kind=Parameter.POSITIONAL_OR_KEYWORD, **default)

        # change __init__ method signature and default values by adding benchmark parameters to base class ones
        signature = inspect.signature(Benchmark.__init__)
        parameters = [v for k, v in signature.parameters.items() if k != 'config']
        parameters += [_parameter(p) for p in target.parameters.values()]
        target.__init__.__signature__ = signature.replace(parameters=parameters)
        target.__init__.__defaults__ = tuple([p.default for p in parameters if p.default is not Parameter.empty])
        # change query method signature by including benchmarks variables only
        parameters = [_parameter(v) for v in target.variables.values()]
        target.query.__signature__ = signature.replace(parameters=parameters)
        return target


benchmark = BenchmarkDecorator()
