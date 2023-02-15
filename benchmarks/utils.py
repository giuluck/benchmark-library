import inspect
import types

import numpy as np


# noinspection PyPep8Naming
class delegates:
    def __init__(self, delegator):
        self.delegator = delegator

    def __call__(self, delegated=None, additional: bool = False):
        # if delegated is None, delegate the __init__ of the delegator to the __init__ of its base (delegated) class
        # otherwise, delegate the delegated function to the delegator function
        if delegated is None:
            delegated, delegator = self.delegator.__base__.__init__, self.delegator.__init__
        else:
            delegated, delegator = delegated, self.delegator
        # retrieve the signature of the delegator function, get its parameters and remove the kwargs
        signature = inspect.signature(delegator)
        params = dict(signature.parameters)
        kwargs = params.pop('kwargs')
        # update the list of parameters with all the parameters from the delegated signature that are not yet present
        params.update({k: v for k, v in inspect.signature(delegated).parameters.items() if k not in params})
        # put the "kwargs" back in the parameters list if necessary
        if additional:
            params['kwargs'] = kwargs
        # update the signature of the delegator function and return it
        # noinspection PyTypeChecker
        delegator.__signature__ = signature.replace(parameters=params.values())
        return self.delegator


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


def merge(*dictionaries: dict) -> dict:
    """Merges multiple dictionaries. In case of key clash, keeps the value of the earliest dictionary.

    :param dictionaries:
        A list of dictionaries to merge.

    :return:
        The merged dictionary.
    """
    output = {}
    for d in dictionaries[::-1]:
        output.update(d)
    return output
