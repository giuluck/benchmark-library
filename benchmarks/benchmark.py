import collections
import re
from abc import abstractmethod, ABCMeta
from typing import Optional, Any, Dict, List

import dill
import numpy as np
from descriptors import classproperty, cachedclassproperty

from datatypes import Variable, Parameter, Constraint
from utils.strings import stringify


class Benchmark(metaclass=ABCMeta):
    """Abstract class for custom benchmarks."""

    Sample = collections.namedtuple('Sample', 'input output')

    # Benchmark specific properties to be overridden
    _package: str
    _variables: List[Variable]
    _parameters: List[Parameter] = []
    _constraints: List[Constraint] = []

    @classproperty
    def _name(self) -> str:
        return ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', self.__name__)).split())

    @cachedclassproperty
    def description(self) -> str:
        """The benchmark description."""
        docstring = self.__doc__.strip().split('\n')
        min_tab = np.min([len(line) - len(line.lstrip()) for line in docstring[1:] if line != ''])
        docstring[1:] = [line[min_tab:] for line in docstring[1:]]
        return '\n'.join(docstring)

    @classproperty
    def variables(self) -> Dict[str, Variable]:
        """The benchmark variables."""
        return {v.name: v for v in self._variables}

    @classproperty
    def parameters(self) -> Dict[str, Parameter]:
        """The benchmark parameters."""
        return {p.name: p for p in self._parameters}

    @classproperty
    def constraints(self):
        """The benchmark constraints."""
        return {c.name: c for c in self._constraints}

    @classmethod
    def describe(cls, brief: bool = True) -> str:
        """Describes the benchmark.

        :param brief:
            Either to print the list of variables, parameters, and other configuration objects by just their name and
            description or with more details; default: 'True'.

        :return:
            A string representing the textual description of the benchmark.
        """
        string = f"{cls._name.upper()}\n\n{cls.description}"

        def _brief(obj: Any) -> str:
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}"

        def _var_full(obj: Variable) -> str:
            dtype = stringify(obj.dtype)
            domain = 'all values' if obj.domain is None else obj.domain
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}\n    - domain: {domain}\n    -   type: {dtype}"

        def _par_full(obj: Parameter) -> str:
            dtype = stringify(obj.dtype)
            domain = 'all values' if obj.domain is None else obj.domain
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}\n    - default: {obj.default}\n    -  domain: {domain}\n    -    type: {dtype}"

        def _cst_full(obj: Constraint) -> str:
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}"

        # define printing pre-processing
        var, par, cst = (_brief, _brief, _brief) if brief else (_var_full, _par_full, _cst_full)
        string += '\n\nVariables:\n' + '\n'.join([var(v) for v in cls._variables])
        if len(cls._parameters) > 0:
            string += '\n\nParameters:\n' + '\n'.join([par(p) for p in cls._parameters])
        if len(cls._constraints) > 0:
            string += '\n\nConstraints:\n' + '\n'.join([cst(c) for c in cls._constraints])
        return string

    @staticmethod
    def load(filepath: str) -> Any:
        """Loads a benchmark instance given a previously serialized dill file.
        Dill is used as backend since it supports serialization of the random number generator (which is not supported
        by json files) as well as functions and other custom data types (which are not supported by pickle files).

        :param filepath:
            The path of the file.

        :return:
            The benchmark instance.
        """
        with open(filepath, "rb") as file:
            return dill.load(file=file)

    def __init__(self, name: Optional[str] = None, seed: int = 42):
        """
        :param name:
            The name of the benchmark instance, or None to use the default one; default: None.

        :param seed:
            The seed for random operations; default: 42.
        """
        self.name: str = self._name.lower() if name is None else name
        """The name of the benchmark instance."""

        self.seed: int = seed
        """The seed for random operations."""

        self.rng: Any = np.random.default_rng(seed=seed)
        """The random number generator for random operations."""

        self.samples: List[Benchmark.Sample] = []
        """The list of samples <input, output> obtained by querying the 'evaluate' function."""

        parameters = self.parameters
        for name, value in self.configuration.items():
            param = parameters[name]
            assert isinstance(value, param.dtype), f"'{name}' should be of type {stringify(param.dtype)}, got " \
                                                   f"value {stringify(value)} of type {stringify(type(value))}"
            assert param.in_domain(value), f"'{name}' domain is {param.domain}, got value {stringify(value)}"

    @property
    def configuration(self):
        return {p: self.__getattribute__(p) for p in self.parameters.keys()}

    @abstractmethod
    def _eval(self, **inputs) -> Any:
        """Evaluates the black-box function.

        :param inputs:
            The input values, which must be indexed by the respective parameter name.

        :return:
            The evaluation of the function in the given input point.
        """
        pass

    def evaluate(self, **inputs) -> Any:
        """Evaluates the black-box function and stores the result in the history of queries.

        :param inputs:
            The input values, which must be indexed by the respective parameter name. If one or more parameters are not
            passed, or if their values are not in the expected ranges, an exception is raised.

        :return:
            The evaluation of the function in the given input point.
        """
        variables = self.variables.copy()
        keys = list(variables.keys())
        # iterate over all the inputs in order to check that they represent valid parameters and valid types/values
        for name, val in inputs.items():
            assert name in variables, f"'{name}' is not a valid input, choose one in {keys}"
            var = variables.pop(name)
            assert isinstance(val, var.dtype), f"'{name}' should be {stringify(var.dtype)}, got {stringify(type(val))}"
            assert var.in_domain(val), f"'{name}' domain is {var.domain}, got value {val}"
        # check that no expected parameter is left without a value
        assert len(variables) == 0, f"No value was provided for parameters {list(variables.keys())}"
        # check global constraints feasibility
        for name, constraint in self.constraints.items():
            msg = f"Global constraint '{name}' ({constraint.domain}) is not satisfied."
            assert constraint.is_feasible(**inputs, **self.configuration), msg
        # evaluate the function, store the results (with variable assignments as well), and eventually return the output
        output = self._eval(**inputs)
        self.samples.append(Benchmark.Sample(input=inputs, output=output))
        for var in self._variables:
            var.assignments.append(inputs[var.name])
        return output

    def serialize(self, filepath: str):
        """Dumps the benchmark configuration into a dill file.
        Dill is used as backend since it supports serialization of the random number generator (which is not supported
        by json files) as well as functions and other custom data types (which are not supported by pickle files).

        :param filepath:
            The path of the file.
        """
        with open(filepath, "wb") as file:
            dill.dump(self, file=file)

    def __repr__(self) -> str:
        parameters = ', '.join([f'{name}={stringify(value)}' for name, value in self.configuration.items()])
        return f"{self.__class__.__name__}(name='{self.name}', seed={self.seed}, {parameters})"
