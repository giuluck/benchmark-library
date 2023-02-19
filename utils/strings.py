import inspect
import types

import numpy as np


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
        case str():
            # surround the string with double quotation marks
            string = f"'{value}'"
        case int() | float() | bool() | bytes():
            # handle default data types
            string = str(value)
        case tuple() | list() | set():
            # process each value recursively
            values = [stringify(v) for v in value]
            # use the pipe (|) symbol to concatenate the elements in case they are all types
            if np.all([isinstance(v, type) for v in value]):
                string = '|'.join(values)
            else:
                # otherwise, use the comma (,) symbol for concatenation, and an appropriate delimiter based on the type
                delimiter = '({})' if isinstance(value, tuple) else ('[{}]' if isinstance(value, list) else '{{}}')
                string = delimiter.format(', '.join(values))
        case types.LambdaType():
            # include the parameters in the lambda representation
            string = f"lambda{inspect.signature(value)}"
        case _:
            # for other complex data types, simply return their name without any other information
            string = value.__name__
    return f'{prefix}{string}{suffix}'
