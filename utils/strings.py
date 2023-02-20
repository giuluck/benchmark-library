import inspect
from types import LambdaType


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
    if value is None:
        return ''
    if isinstance(value, dict):
        elements = [stringify(k) + ': ' + stringify(v) for k, v in value.items()]
        string = f"{{{', '.join(elements)}}}"
    elif isinstance(value, set | list | tuple):
        delimiter = '({})' if isinstance(value, tuple) else ('[{}]' if isinstance(value, list) else '{{{}}}')
        string = delimiter.format(', '.join([stringify(v) for v in value]))
    elif isinstance(value, LambdaType):
        # include the parameters in the function representation
        string = f"{value.__name__}{inspect.signature(value)}"
    elif hasattr(value, '__name__'):
        string = value.__name__
    else:
        string = f"{value!r}"
    return f'{prefix}{string}{suffix}'
