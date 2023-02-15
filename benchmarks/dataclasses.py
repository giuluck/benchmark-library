from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from benchmarks.utils import stringify, merge


@dataclass
class _AbstractDataClass:
    """Abstract template for a Data Class"""

    @staticmethod
    def _get_signature(description: Optional[str] = None,
                       dtype: Optional[str] = None,
                       domain: Optional[str] = None,
                       default: Optional[str] = None,
                       **modules):
        """General-purpose parsing utility."""
        scope = merge(modules, locals(), globals())
        signature = dict()
        if description is not None:
            signature['description'] = description
        if dtype is not None:
            signature['dtype'] = eval(dtype, scope)
        if domain is not None:
            signature['domain'] = domain
        if default is not None:
            signature['default'] = eval(default, scope) if isinstance(default, str) else default
        return signature

    name: str
    """The name of the instance, which must follow the python requirements about variables' names."""

    domain: Optional[str] = field(default=None, kw_only=True)
    """An optional textual description of the domain; default: None."""

    description: Optional[str] = field(default=None, kw_only=True)
    """An optional textual description of the instance; default: None."""


@dataclass
class _AbstractVariable(_AbstractDataClass):
    """Abstract template for a Variable"""
    dtype: type = field(default=object, kw_only=True)
    """The instance type; default: object."""

    in_domain: Callable = field(default=lambda v: True, kw_only=True)
    """A function f(x) -> bool which defines whether the variable domain; default: f(x) -> True."""


@dataclass
class Variable(_AbstractVariable):
    """A benchmark variable."""

    @staticmethod
    def parse(name: str,
              dtype: Optional[str] = None,
              domain: Optional[str] = None,
              description: Optional[str] = None,
              **modules) -> Any:
        """Creates an instance by parsing a text configuration.

        :param name:
            The name of the instance.

        :param dtype:
            The (optional) instance type. If None, the default type is used; default: None.

        :param domain:
            The (optional) instance domain. If None, the default domain is used; default: None.

        :param description:
            The (optional) instance description. If None, the default description is used; default: None.

        :param modules:
            Dynamically loaded modules necessary in the eval function.

        :return:
            The parsed instance.
        """
        signature = _AbstractDataClass._get_signature(
            dtype=dtype,
            domain=domain,
            description=description,
            **modules
        )
        if 'domain' in signature:
            signature['in_domain'] = eval(f"lambda v: {domain}", merge(modules, locals(), globals()))
        return Variable(name=name, **signature)

    def __repr__(self):
        return self.name if self.description is None else f"{self.name}: {self.description}"


# noinspection PyDataclass
@dataclass
class Parameter(_AbstractVariable):
    default: Any
    """The default value of the instance."""

    @staticmethod
    def parse(name: str,
              default: Any,
              dtype: Optional[str] = None,
              domain: Optional[str] = None,
              description: Optional[str] = None,
              **modules) -> Any:
        """Creates an instance by parsing a text configuration.

        :param name:
            The name of the instance.

        :param default:
            The instance default value.

        :param dtype:
            The (optional) instance type. If None, the default type is used; default: None.

        :param domain:
            The (optional) instance domain. If None, the default domain is used; default: None.

        :param description:
            The (optional) instance description. If None, the default description is used; default: None.

        :param modules:
            Dynamically loaded modules necessary in the eval function.

        :return:
            The parsed instance.
        """
        signature = _AbstractDataClass._get_signature(
            default=default,
            dtype=dtype,
            domain=domain,
            description=description,
            **modules
        )
        # check that default value is in domain if there is a domain
        if 'domain' in signature:
            default = signature['default']
            in_domain = eval(f"lambda v: {domain}", merge(modules, locals(), globals()))
            assert in_domain(default), f"default value {stringify(default)} is out of domain"
            signature['in_domain'] = in_domain
        return Parameter(name=name, **signature)

    def __repr__(self):
        description = self.name if self.description is None else f"{self.name}: {self.description}"
        return f"{description}; default = {stringify(self.default)}"


# noinspection PyDataclass
@dataclass
class Constraint(_AbstractDataClass):
    is_feasible: Callable
    """A function f(**variables, **params) -> True which defines whether the constraint is satisfied or not."""

    @staticmethod
    def parse(*inputs, name: str, domain: Optional[str] = None, description: Optional[str] = None, **modules) -> Any:
        """Creates an instance by parsing a text configuration.
        :param inputs:
            The names of all the input parameters and variables in the benchmark instance.

        :param name:
            The name of the instance.

        :param domain:
            The (optional) instance domain. If None, the default domain is used; default: None.

        :param description:
            The (optional) instance description. If None, the default description is used; default: None.

        :param modules:
            Dynamically loaded modules necessary in the eval function.

        :return:
            The parsed instance.
        """
        signature = _AbstractDataClass._get_signature(domain=domain, description=description, **modules)
        if 'domain' in signature:
            inputs = ', '.join(inputs)
            signature['is_feasible'] = eval(f"lambda {inputs}: {domain}", merge(modules, locals(), globals()))
        return Constraint(name=name, **signature)

    def __repr__(self):
        return self.domain if self.description is None else self.description
