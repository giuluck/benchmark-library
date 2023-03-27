import re
from abc import abstractmethod, ABC
from typing import List, Any, Optional, Dict, Callable

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from descriptors import classproperty, cachedclassproperty

from model.datatypes import Variable, Parameter, Constraint, Sample, Metric
from model.structure import Structure
from model.utils import stringify


def querymethod(target: Callable) -> Callable:
    """Decorator for benchmark 'query' abstract method."""

    def query(self, **inputs: Any) -> Any:
        variables = {v.name: v for v in self.structure.variables}
        # iterate over all the inputs in order to check that they represent valid parameters and valid types/values
        for name, val in inputs.items():
            assert name in variables, f"'{name}' is not a valid input, choose one in {list(self.structure.variables)}"
            var = variables.pop(name)
            assert var.check_dtype(val), f"'{name}' should be {stringify(var.dtype)}, got {stringify(type(val))}"
            assert var.check_domain(val), f"'{name}' domain is {var.domain}, got value {val}"
        # check that no expected parameter is left without a value
        assert len(variables) == 0, f"No value was provided for variables {list(variables.keys())}"
        # create new dictionary containing both input variables and parameter configuration
        check = {**inputs, **self.configuration}
        # check global constraints feasibility
        for cst in self.structure.constraints:
            desc = "" if cst.description is None else f"{cst.description}; "
            assert cst.check(**{i: check[i] for i in cst.inputs}), f"{desc}global constraint '{cst.name}' not satisfied"
        # evaluate the function, store the results, and eventually return the output
        output = target(self, **inputs)
        self._samples.append(Sample(inputs=inputs, output=output))
        return output

    return query


class Benchmark(ABC):
    """Abstract class for custom benchmarks."""

    # ------------------------------------- ABSTRACT BENCHMARK OPERATIONS ----------------------------------------------
    #   - these properties and operations must be overridden by each benchmark child class
    #   - 'build' defines the benchmark structure (i.e., its variables, parameters, constraints, and metrics)
    #   - 'query' defines the benchmark sampling function (it must be decorated with the 'querymethod' decorator)
    #   - '__init__' defines the benchmark initialization (it is best to override it in order to set its parameters)
    @staticmethod
    @abstractmethod
    def build(structure: Structure):
        """Builds the structure of the benchmark.

        :param structure:
            A Structure object which can be used to add variables, parameters, constraints, and metrics.
        """
        pass

    @querymethod
    @abstractmethod
    def query(self, **inputs: Any) -> Any:
        """Evaluates the black-box function.

        :param inputs:
            The input values, containing both variables and parameters instances, indexed by name.

        :return:
            The evaluation of the function in the given input point.
        """
        pass

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None, **configuration):
        """
        :param name:
            The name of the benchmark instance, or None to use the default one; default: None.

        :param seed:
            The seed for random operations, or None if the benchmark expects no random operation; default: None.

        :param configuration:
            The benchmark-specific parameter values. If a parameter is not explicitly passed, its default value is used.
        """
        self._seed: Optional[int] = seed
        self._rng: np.random.Generator = np.random.default_rng(seed=seed)
        self._name: str = self.alias.lower() if name is None else name
        self._configuration: Dict[str, Any] = {}
        self._samples: List[Sample] = []
        # check parameters configuration consistency
        for par in self.structure.parameters:
            if par.name in configuration:
                value = configuration[par.name]
                assert par.check_dtype(value), f"'{par.name}' should be of type {par.dtype.__name__}, got " \
                                               f"value {stringify(value)} of type {type(value).__name__}"
                assert par.check_domain(value), f"'{par.name}' domain is {par.domain}, got value {stringify(value)}"
            else:
                value = par.default
            self._configuration[par.name] = value

    # ------------------------------- SHARED BENCHMARK OPERATIONS AND PROPERTIES ---------------------------------------
    #   - these properties and operations are automatically generated from the benchmark child class
    #   - for some properties that might be computationally intensive or frequently computed, their values can be
    #     stored at class level leveraging the @cachedclassproperty decorator
    @cachedclassproperty
    def structure(self) -> Structure:
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
            docstring[1:] = [line[min_tab:].rstrip() for line in docstring[1:]]
            description = '\n'.join(docstring)
        # build a new structure object with the given alias and description, and pass it to the '_build' routine
        structure = Structure(alias=alias, description=description)
        self.build(structure=structure)
        return structure

    @classproperty
    def alias(self) -> str:
        """The benchmark alias."""
        return str(self.structure.alias)

    @classproperty
    def description(self) -> str:
        """The benchmark description."""
        return str(self.structure.description)

    @classproperty
    def variables(self) -> Dict[str, str]:
        """The benchmark variables."""
        return {v.name: '' if v.description is None else v.description for v in self.structure.variables}

    @classproperty
    def parameters(self) -> Dict[str, str]:
        """The benchmark parameters."""
        return {p.name: '' if p.description is None else p.description for p in self.structure.parameters}

    @classproperty
    def constraints(self) -> Dict[str, str]:
        """The benchmark constraints."""
        return {c.name: '' if c.description is None else c.description for c in self.structure.constraints}

    @classproperty
    def metrics(self) -> Dict[str, str]:
        """The benchmark metrics."""
        return {m.name: '' if m.description is None else m.description for m in self.structure.metrics}

    @classmethod
    def describe(cls, brief: bool = True) -> str:
        """Describes the benchmark.

        :param brief:
            Either to print the list of variables, parameters, and other configuration objects by just their name and
            description or with more details; default: 'True'.

        :return:
            A string representing the textual description of the benchmark.
        """
        string = f"{cls.structure.alias.upper()}\n\n{cls.structure.description}"

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

        def _mtr_full(obj: Metric) -> str:
            description = obj.name if obj.description is None else f"{obj.name}: {obj.description}"
            return f"  * {description}"

        # define printing pre-processing
        var, par, cst, mtr = (_brief, _brief, _brief, _brief) if brief else (_var_full, _par_full, _cst_full, _mtr_full)
        if len(cls.structure.variables) > 0:
            string += '\n\nVariables:\n' + '\n'.join([var(v) for v in cls.structure.variables])
        if len(cls.structure.parameters) > 0:
            string += '\n\nParameters:\n' + '\n'.join([par(p) for p in cls.structure.parameters])
        if len(cls.structure.constraints) > 0:
            string += '\n\nConstraints:\n' + '\n'.join([cst(c) for c in cls.structure.constraints])
        if len(cls.structure.metrics) > 0:
            string += '\n\nMetrics:\n' + '\n'.join([mtr(m) for m in cls.structure.metrics])
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
    #   - instance properties and operations are those tied to a benchmark instance created by the end users
    @property
    def name(self) -> str:
        """The name of the benchmark instance."""
        return str(self._name)

    @property
    def seed(self) -> Optional[int]:
        """The seed for random operations."""
        return self._seed

    @property
    def samples(self) -> List[Sample]:
        """The list of samples <inputs, output> obtained by querying the 'evaluate' function."""
        return list(self._samples)

    @property
    def configuration(self) -> Dict[str, Any]:
        """The benchmark-specific parameter values."""
        return dict(self._configuration)

    def evaluate(self, sample: Optional[Sample] = None) -> pd.Series:
        """Computes all the benchmark metrics on the given sample.

        :param sample:
            Either a sample instance, or None to compute the metrics on the last queried sample.

        :return:
            A Series object where each metric (name) is associated to its computed value.
        """
        if sample is None:
            assert len(self._samples) > 0, "No samples to evaluate in the benchmark history"
            sample = self._samples[-1]
        metrics = {metric.name: metric.evaluate(sample) for metric in self.structure.metrics}
        return pd.Series(metrics)

    def history(self, plot: bool | Dict[str, Any] = False) -> pd.DataFrame:
        """Evaluates all the queried samples and optionally plot their values.

        :param plot:
            Either a boolean stating whether to plot the metrics history or not, or a dictionary of parameters to be
            passed to the plt.figure() method. Each metric is plotted in a different figure.

        :return:
            A dataframe where each column represents a metric, and each row a queried sample.
        """
        history = pd.DataFrame([self.evaluate(sample) for sample in self._samples])
        plot = dict(figsize=(16, 9), tight_layout=True) if plot is True else plot
        if isinstance(plot, dict):
            index = history.index + 1
            for name, metric in self.structure.metrics.items():
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
        parameters = ', '.join([f'{name}={stringify(value)}' for name, value in self._configuration.items()])
        return f"{self.__class__.__name__}(name='{self._name}', seed={self._seed}, {parameters})"
