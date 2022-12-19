import types
from typing import TypeVar

# var types and generics
num = int | float
T = TypeVar("T")
N = TypeVar("N", int, float)


def to_string(value: T) -> str:
    """Custom to_string method to handle certain data types.

    :param value:
        Any object.

    :return:
        The string version of the given object.
    """
    match value:
        case str():
            # surround the string with double quotation marks
            return f"'{value}'"
        case type():
            # remove <class '...'> from type string representation
            return str(value)[8:-2]
        case types.FunctionType():
            # remove <function '...' at 0x0123456789ABCDEF> from type string representation
            return str(value)[10:-23]
        case _:
            return str(value)


def none_string(value: T, prefix: str = "", suffix: str = "") -> str:
    """Handles custom textual descriptions when None values can be passed.

    :param value:
        Any object, or None.

    :param prefix:
        A string to be prepended to the value in case it is not None; default: "".

    :param suffix:
        A string to be appended to the value in case it is not None; default: "".

    :return:
        Either an empty string if value is None, or a string of type <prefix><value><suffix>
    """
    return "" if value is None else f"{prefix}{value}{suffix}"
