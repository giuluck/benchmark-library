import re
from abc import abstractmethod, ABC
from typing import List, Any, Optional, Dict

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from descriptors import classproperty, cachedclassproperty

from model.datatypes import Variable, Parameter, Constraint, Metric, Sample
from model.structure import Structure
from model.utils import stringify


class Benchmark(ABC):
    """Abstract class for custom benchmarks."""

    # ------------------------------------- ABSTRACT BENCHMARK OPERATIONS ----------------------------------------------
    #   - these properties and operations must be overridden by each benchmark child class
    #   - '_structure' defines the benchmark structure (i.e., its variables, parameters, constraints, and metrics)
    #   - '_query' defines the benchmark sampling function
    @staticmethod
    @abstractmethod
    def _build(structure: Structure):
        """Builds the structure of the benchmark.

        :param structure:
            A Structure object which can be used to add variables, parameters, constraints, and metrics.
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

    # ------------------------------- SHARED BENCHMARK OPERATIONS AND PROPERTIES ---------------------------------------
    #   - these properties and operations are automatically generated from the benchmark child class
    #   - for some properties that might be computationally intensive or frequently computed, their values can be
    #     stored at class level leveraging the @cachedclassproperty decorator
    @cachedclassproperty
    def _structure(self) -> Structure:
        """Calls the '_build' abstract method in order to build and store the internal benchmark structure."""
        # retrieve alias from class name
        alias = ' '.join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', self.__name__)).split())
        # retrieve description from docstring
        if self.__doc__ is None:
            description = ""
        elif '\n' not in self.__doc__:
            description = self.__doc__
        else:
            docstring = self.__doc__.strip().split('\n')
            min_tab = np.min([len(line) - len(line.lstrip()) for line in docstring[1:] if line != ''])
            docstring[1:] = [line[min_tab:] for line in docstring[1:]]
            description = '\n'.join(docstring)
        # build a new structure object with the given alias and description, and pass it to the '_build' routine
        structure = Structure(alias=alias, description=description)
        self._build(structure=structure)
        return structure

    @classproperty
    def alias(self) -> str:
        """The benchmark alias."""
        return self._structure.alias

    @classproperty
    def description(self) -> str:
        """The benchmark description."""
        return self._structure.description

    @classproperty
    def variables(self) -> List[Variable]:
        """The benchmark variables."""
        return self._structure.variables

    @classproperty
    def parameters(self) -> List[Parameter]:
        """The benchmark parameters."""
        return self._structure.parameters

    @classproperty
    def constraints(self) -> List[Constraint]:
        """The benchmark constraints."""
        return self._structure.constraints

    @classproperty
    def metrics(self) -> List[Metric]:
        """The benchmark metrics."""
        return self._structure.metrics

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
        string += '\n\nVariables:\n' + '\n'.join([var(v) for v in cls.variables])
        if len(cls.parameters) > 0:
            string += '\n\nParameters:\n' + '\n'.join([par(p) for p in cls.parameters])
        if len(cls.constraints) > 0:
            string += '\n\nConstraints:\n' + '\n'.join([cst(c) for c in cls.constraints])
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

    # ------------------------------ INSTANCE BENCHMARK OPERATIONS AND PROPERTIES --------------------------------------
    #   - instance properties and operations are those defined by the end users rather than the benchmark developers
    #   - the '__init__' and 'query' operations are capable of handling any type of input, still it could be beneficial
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

        self.configuration: Dict[str, Any] = {}
        """The benchmark-specific parameter values."""

        # check parameters configuration consistency
        for par in self.parameters:
            if par.name in configuration:
                value = configuration[par.name]
                assert par.check_dtype(value), f"'{par.name}' should be of type {par.dtype.__name__}, got " \
                                               f"value {stringify(value)} of type {type(value).__name__}"
                assert par.check_domain(value), f"'{par.name}' domain is {par.domain}, got value {stringify(value)}"
            else:
                value = par.default
            self.configuration[par.name] = value

        # add properties for configuration parameters
        def set_property(field: str):
            def fget(inst):
                return inst.configuration[field]

            def fset(inst, val):
                inst.configuration[field] = val

            setattr(self.__class__, field, property(fget=fget, fset=fset, fdel=None))

        for par in self.configuration.keys():
            set_property(par)

    def query(self, **inputs) -> Any:
        """Evaluates the black-box function and stores the result in the history of queries.

        :param inputs:
            The input values, which must be indexed by the respective parameter name. If one or more parameters are not
            passed, or if their values are not in the expected ranges, an exception is raised.

        :return:
            The evaluation of the function in the given input point.
        """
        variables = {v.name: v for v in self.variables}
        # iterate over all the inputs in order to check that they represent valid parameters and valid types/values
        for name, val in inputs.items():
            assert name in variables, f"'{name}' is not a valid input, choose one in {list(self.variables.keys())}"
            var = variables.pop(name)
            assert var.check_dtype(val), f"'{name}' should be {stringify(var.dtype)}, got {stringify(type(val))}"
            assert var.check_domain(val), f"'{name}' domain is {var.domain}, got value {val}"
        # check that no expected parameter is left without a value
        assert len(variables) == 0, f"No value was provided for variables {list(variables.keys())}"
        # check global constraints feasibility
        for cst in self.constraints:
            desc = "" if cst.description is None else f"{cst.description}; "
            assert cst.check(**inputs, **self.configuration), f"{desc}global constraint '{cst.name}' not satisfied"
        # evaluate the function, store the results, and eventually return the output
        output = self._query(**inputs, **self.configuration)
        self.samples.append(Sample(inputs=inputs, output=output))
        return output

    def evaluate(self, sample: Optional[Sample] = None) -> pd.Series:
        """Computes all the benchmark metrics on the given sample.

        :param sample:
            Either a sample instance, or None to compute the metrics on the last queried sample.

        :return:
            A Series object where each metric (name) is associated to its computed value.
        """
        if sample is None:
            assert len(self.samples) > 0, "No samples to evaluate in the benchmark history"
            sample = self.samples[-1]
        metrics = {name: metric(sample) for name, metric in self.metrics.items()}
        return pd.Series(metrics)

    def history(self, plot: bool | Dict[str, Any] = False) -> pd.DataFrame:
        """Evaluates all the queried samples and optionally plot their values.

        :param plot:
            Either a boolean stating whether to plot the metrics history or not, or a dictionary of parameters to be
            passed to the plt.figure() method. Each metric is plotted in a different figure.

        :return:
            A dataframe where each column represents a metric, and each row a queried sample.
        """
        history = pd.DataFrame([self.evaluate(sample) for sample in self.samples])
        plot = dict(figsize=(16, 9), tight_layout=True) if plot is True else plot
        if isinstance(plot, dict):
            index = history.index + 1
            for name, metric in self.metrics.items():
                plt.figure(**plot)
                plt.plot(index, history[name])
                plt.title(metric.description.title() if metric.description is not None else name.title())
                ticks = np.array([int(tick) for tick in plt.xticks()[0] if 0 < int(tick) <= len(index)])
                plt.xticks(ticks - ticks[0] // 2)
                plt.xlim(0, len(index) + 1)
                plt.ylabel(name)
                plt.show()
        return history

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
