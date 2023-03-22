import inspect
from inspect import Parameter
from types import LambdaType, FunctionType, MethodType
from typing import Any


def stringify(value, prefix: str = '', suffix: str = '') -> str:
    """Custom method to handle convert certain data types to strings.

    :param value:
        Any object.

    :param prefix:
        A prefix in the final string version; default = ''.

    :param suffix:
        A suffix in the final string version; default = ''.

    :return:
        The string version of the given object.
    """
    match value:
        case None:
            return ''
        case dict():
            elements = [stringify(k) + ': ' + stringify(v) for k, v in value.items()]
            string = f"{{{', '.join(elements)}}}"
        case set() | list() | tuple():
            delimiter = '({})' if isinstance(value, tuple) else ('[{}]' if isinstance(value, list) else '{{{}}}')
            string = delimiter.format(', '.join([stringify(v) for v in value]))
        case LambdaType() | FunctionType() | MethodType():
            # include the parameters in the function representation
            string = f"{value.__name__}{inspect.signature(value)}"
        case _:
            string = value.__name__ if hasattr(value, '__name__') else f"{value!r}"
    return f'{prefix}{string}{suffix}'


class BenchmarkDecorator:
    """Decorator for a benchmark instance.
    It changes the signature of the '__init__' and 'query'  methods in order to include the benchmark-specific inputs.
    Even though the decorator takes care of changing the input signature of the '__init__' and the 'query' methods,
    overriding them with the correct parameter names is helpful for autocompletion in many IDEs. Similarly, the
    decorator also takes care of the default values and annotations, but adding them will be beneficial for static type
    checking and signatures. Therefore, the decorator can be used to find the wanted trade-off between detailed static
    API and boilerplate/maintenance costs.
    Finally, it must be noted that the decorator needs to check the list of variables and parameters, hence it breaks
    late initialization by accessing late-initialized class properties before at import time.
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
        # (import benchmark here to avoid circular dependencies due to stringify)
        from model.benchmark import Benchmark
        signature = inspect.signature(Benchmark.__init__)
        parameters = [v for k, v in signature.parameters.items() if k != 'configuration']
        parameters += [_parameter(p) for p in target.parameters]
        target.__init__.__signature__ = signature.replace(parameters=parameters)
        target.__init__.__defaults__ = tuple([p.default for p in parameters if p.default is not Parameter.empty])
        # change query method signature by including benchmarks variables only
        parameters = [Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD)]
        parameters += [_parameter(v) for v in target.variables]
        target.query.__signature__ = signature.replace(parameters=parameters)
        return target


benchmark = BenchmarkDecorator()
