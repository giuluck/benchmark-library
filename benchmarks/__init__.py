import re
from abc import abstractmethod, ABC
from typing import Optional, Any, Dict, List

import dill
import numpy as np
from descriptors import classproperty, cachedclassproperty

from datatypes import Variable, Parameter, Constraint, DataType, Sample
from datatypes.metrics import Metric
from utils.strings import stringify


class Benchmark(ABC):
    """Abstract class for custom benchmarks."""

    # BENCHMARK-SPECIFIC PROPERTIES AND OPERATIONS
    #   - these properties and operations must be overridden by each benchmark child class
    #   - the '_structure' property is not formally defined as such but as a function to allow for late initialization,
    #     still, it can be seen as a property '_structure: Callable[[], List[DataType]]' thus defined with via a
    #     non-parametric lambda if preferred
    @staticmethod
    @abstractmethod
    def _structure() -> List[DataType]:
        """Defines the structure of a benchmark.

        :return:
            A list of variables, parameters, constraints, and metrics which define the benchmark structure.
        """
        pass

    @staticmethod
    @abstractmethod
    def _query(**inputs) -> Any:
        """Evaluates the black-box function.

        :param inputs:
            The input values, containing both variables and parameters instances, indexed by name.

        :return:
            The evaluation of the function in the given input point.
        """
        pass

    # SHARED BENCHMARK PROPERTIES AND OPERATIONS
    #   - these properties and operations are automatically generated from the benchmark child class
    #   - for some properties that might be computationally intensive or frequently computed, their values can be
    #     stored at class level leveraging the @cachedclassproperty decorator
    @cachedclassproperty
    def _unpacked_structure(self) -> Dict[str, Any]:
        """Calls the benchmark-specific '_structure' property once and unpacks it in a cached structure dictionary.
        Additionally, it checks benchmark consistency."""
        # retrieve benchmark-specific structure
        structure = self._structure()
        # check for the presence of at least one variable and no name clashes
        names = sorted([obj.name for obj in structure])
        assert np.any([isinstance(obj, Variable) for obj in structure]), "Benchmarks should have at least one variable"
        for n1, n2 in zip(names[:-1], names[1:]):
            assert n1 != n2, f"Data types cannot have duplicate names, got duplicate name '{n1}'"
        # retrieve alias from class name
        alias = ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', self.__name__)).split())
        # retrieve description from docstring
        docstring = self.__doc__.strip().split('\n')
        min_tab = np.min([len(line) - len(line.lstrip()) for line in docstring[1:] if line != ''])
        docstring[1:] = [line[min_tab:] for line in docstring[1:]]
        description = '\n'.join(docstring)
        # retrieve objects from structure
        objects = {namespace: dict() for namespace in ['variables', 'parameters', 'constraints', 'metrics']}
        for obj in structure:
            match obj:
                case Variable():
                    namespace = 'variables'
                case Parameter():
                    namespace = 'parameters'
                case Constraint():
                    namespace = 'constraints'
                case Metric():
                    namespace = 'metrics'
                case _:
                    raise AssertionError(f"Unsupported object type {type(obj).__name__} in benchmark structure")
            objects[namespace][obj.name] = obj
        return {'alias': alias, 'description': description, **objects}

    @classproperty
    def alias(self) -> str:
        """The benchmark alias."""
        return self._unpacked_structure['alias']

    @classproperty
    def description(self) -> str:
        """The benchmark description."""
        return self._unpacked_structure['description']

    @classproperty
    def variables(self) -> Dict[str, Variable]:
        """The benchmark variables."""
        return self._unpacked_structure['variables']

    @classproperty
    def parameters(self) -> Dict[str, Parameter]:
        """The benchmark parameters."""
        return self._unpacked_structure['parameters']

    @classproperty
    def constraints(self):
        """The benchmark constraints."""
        return self._unpacked_structure['constraints']

    @classproperty
    def metrics(self):
        """The benchmark metrics."""
        return self._unpacked_structure['metrics']

    @classmethod
    def describe(cls, brief: bool = True) -> str:
        """Describes the benchmark.

        :param brief:
            Either to print the list of variables, parameters, and other configuration objects by just their name and
            description or with more details; default: 'True'.

        :return:
            A string representing the textual description of the benchmark.
        """
        string = f"{cls.alias.upper()}\n\n{cls.description}"

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
        string += '\n\nVariables:\n' + '\n'.join([var(v) for v in cls.variables.values()])
        if len(cls.parameters) > 0:
            string += '\n\nParameters:\n' + '\n'.join([par(p) for p in cls.parameters.values()])
        if len(cls.constraints) > 0:
            string += '\n\nConstraints:\n' + '\n'.join([cst(c) for c in cls.constraints.values()])
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

    # INSTANCE PROPERTIES AND OPERATIONS
    #   - instance properties and operations are those defined by the end users rather than the benchmark developers
    #   - the '__init__' and 'query' operations are capable of handling any tipe of input, still it could be beneficial
    #     for the user to re-define them by adding the benchmark-specific parameters in order to allow for IDE hints,
    #     autocompletion, and static type checking
    def __init__(self, name: Optional[str] = None, seed: int = 42, **configuration):
        """
        :param name:
            The name of the benchmark instance, or None to use the default one; default: None.

        :param seed:
            The seed for random operations; default: 42.

        :param configuration:
            The benchmark-specific parameter values. If a parameter is not explicitly passed, its default value is used.
        """
        self._rng: Any = np.random.default_rng(seed=seed)
        """The random number generator to be used for (internal only) random operations."""

        self.name: str = self.alias.lower() if name is None else name
        """The name of the benchmark instance."""

        self.seed: int = seed
        """The seed for random operations."""

        self.samples: List[Sample] = []
        """The list of samples <inputs, output> obtained by querying the 'evaluate' function."""

        self.configuration: Dict[str, Any] = {n: configuration.get(n, p.default) for n, p in self.parameters.items()}
        """The benchmark-specific parameter values."""

        # check parameters configuration consistency
        for name, value in self.configuration.items():
            param = self.parameters[name]
            assert isinstance(value, param.dtype), f"'{name}' should be of type {param.dtype.__name__}, got " \
                                                   f"value {stringify(value)} of type {type(value).__name__}"
            assert param.in_domain(value), f"'{name}' domain is {param.domain}, got value {stringify(value)}"

        # add properties for configuration parameters
        def set_property(field: str):
            def fget(inst):
                return inst.configuration[field]

            def fset(inst, val):
                inst.configuration[field] = val

            setattr(self.__class__, field, property(fget=fget, fset=fset, fdel=None))

        for param in self.configuration.keys():
            set_property(param)

    def query(self, **inputs) -> Any:
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
            msg = f"{constraint.description.capitalize()}; global constraint '{constraint.name}' not satisfied"
            assert constraint.is_satisfied(**inputs, **self.configuration), msg
        # evaluate the function, store the results, and eventually return the output
        output = self._query(**inputs, **self.configuration)
        self.samples.append(Sample(inputs=inputs, output=output))
        return output

    # def evaluate(self, sample: Optional[Sample] = None) -> pd.Series:
    #     if sample is None:
    #         assert len(self.samples) > 0, "No samples to evaluate in the benchmark history"
    #         sample = self.samples[-1]
    #     metrics = {m for m in self}

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
