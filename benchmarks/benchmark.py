import collections
import importlib.resources
import re
from typing import Optional, Any, Dict, List

import dill
import numpy as np
import yaml
from descriptors import cachedclassproperty, classproperty

from benchmarks.dataclasses import Variable, Parameter, Constraint
from benchmarks.utils import stringify


class Benchmark:
    """Abstract class for custom benchmarks."""

    Sample = collections.namedtuple('Sample', 'input output')

    _package: str
    """The benchmark package name."""

    @cachedclassproperty
    def _name(self) -> str:
        return ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', self.__name__)).split())

    @cachedclassproperty
    def _configuration(self) -> Dict[str, Any]:
        """Lazily imports the configuration of the benchmark from the yaml configuration."""
        # load yaml properties from config file
        with importlib.resources.open_binary(f'benchmarks.{self._package}', 'config.yaml') as file:
            properties = yaml.safe_load(file)
        # handle input fields
        assert 'variables' in properties, "Missing variables section in config.yaml"
        properties['parameters'] = properties['parameters'] if 'parameters' in properties else {}
        properties['constraints'] = properties['constraints'] if 'constraints' in properties else {}
        properties['description'] = properties['description'] if 'description' in properties else ''
        properties['import'] = properties['import'] if 'import' in properties else []
        # handle imports
        modules = {}
        for module in (properties['import'] if 'import' in properties else []):
            module = module.split(' as ')
            module, alias = (module[0], module[0]) if len(module) == 1 else module
            modules[alias] = __import__(module)
        # parse variables and parameters
        variables, parameters, constraints = {}, {}, {}
        for name, attributes in properties['variables'].items():
            variables[name] = Variable.parse(name=name, **attributes, **modules)
        for name, attributes in properties['parameters'].items():
            parameters[name] = Parameter.parse(name=name, **attributes, **modules)
        # parse constraints
        inputs = [*variables.keys(), *parameters.keys()]
        for name, attributes in properties['constraints'].items():
            attributes = {'domain': attributes} if isinstance(attributes, str) else attributes
            constraints[name] = Constraint.parse(*inputs, name=name, **attributes, **modules)
        # return configuration dictionary
        return {
            'variables': variables,
            'parameters': parameters,
            'constraints': constraints,
            'description': properties['description']
        }

    @classproperty
    def description(self) -> str:
        """The benchmark description."""
        return self._configuration['description']

    @classproperty
    def variables(self) -> Dict[str, Variable]:
        """The benchmark variables."""
        return self._configuration['variables']

    @classproperty
    def parameters(self) -> Dict[str, Parameter]:
        """The benchmark parameters."""
        return self._configuration['parameters']

    @classproperty
    def constraints(self):
        """The benchmark constraints."""
        return self._configuration['constraints']

    @classmethod
    def describe(cls, brief: bool = True) -> str:
        """Describes the benchmark.

        :param brief:
            Either to print the list of variables, parameters, and other configuration objects by just their name and
            description or with more details; default: 'True'.

        :return:
            A string representing the textual description of the benchmark.
        """

        def _brief(obj: Any) -> str:
            return f"  * {obj}"

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
            domain = 'all values' if obj.domain is None else obj.domain
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}\n    - domain: {domain}"

        # define printing pre-processing
        var, par, cst = (_brief, _brief, _brief) if brief else (_var_full, _par_full, _cst_full)
        variables = 'Variables:\n' + '\n'.join([var(v) for v in cls.variables.values()])
        parameters = 'Parameters:\n' + '\n'.join([par(p) for p in cls.parameters.values()])
        constraints = 'Constraints:\n' + '\n'.join([cst(c) for c in cls.constraints.values()])
        return '\n\n'.join([cls._name.upper(), cls.description, variables, parameters, constraints])

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
            dump = dill.load(file=file)
        return dump

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

    @property
    def configuration(self):
        return {p: self.__getattribute__(p) for p in self.parameters.keys()}

    def __repr__(self) -> str:
        parameters = ', '.join([f'{name}={stringify(value)}' for name, value in self.configuration.items()])
        return f"{self.__class__.__name__}(name='{self.name}', seed={self.seed}, {parameters})"

    def _eval(self, **inputs) -> Any:
        """Evaluates the black-box function.

        :param inputs:
            The input values, which must be indexed by the respective parameter name.

        :return:
            The evaluation of the function in the given input point.
        """
        raise NotImplementedError("Please implement abstract method '_eval'")

    def evaluate(self, **inputs) -> Any:
        """Evaluates the black-box function and stores the result in the history of queries.

        :param inputs:
            The input values, which must be indexed by the respective parameter name. If one or more parameters are not
            passed, or if their values are not in the expected ranges, an exception is raised.

        :return:
            The evaluation of the function in the given input point.
        """
        variables = self.variables.copy()
        # iterate over all the inputs in order to check that they represent valid parameters and valid types/values
        for name, val in inputs.items():
            assert name in variables, f"'{name}' is not a valid input, choose one in {list(self.variables.keys())}"
            var = variables.pop(name)
            assert isinstance(val, var.dtype), f"'{name}' should be {stringify(var.dtype)}, got {stringify(type(val))}"
            assert var.in_domain(val), f"'{name}' domain is {var.domain}, got value {val}"
        # check that no expected parameter is left without a value
        assert len(variables) == 0, f"No value was provided for parameters {list(variables.keys())}"
        # check global constraints feasibility
        for name, constraint in self.constraints.items():
            msg = f"Global constraint '{name}' ({constraint.domain}) is not satisfied."
            assert constraint.is_feasible(**inputs, **self.configuration), msg
        # evaluate the function, store the results, and eventually return the output
        output = self._eval(**inputs)
        self.samples.append(Benchmark.Sample(input=inputs, output=output))
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
