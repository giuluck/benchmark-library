import inspect
from types import LambdaType, FunctionType, MethodType


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
