import json
import os
import pickle
from typing import Dict, List, Any, Optional

import dill
import pandas as pd

from parameters import Parameter
from utils import T, to_string


class Benchmark:
    """Abstract class for custom benchmarks."""

    ALIAS: str
    """A short alias for the benchmark."""

    DESCRIPTION: str
    """A textual description of the benchmark."""

    PARAMETERS: List[Parameter]
    """The list of parameters of the benchmark."""

    @classmethod
    def load(cls, filepath: str, filetype: Optional[str] = None) -> T:
        """Loads a benchmark instance given a previously serialized pickle file.

        :param filepath:
            The path of the pickle file.

        :param filetype:
            The type of encoding, or None to infer it from the filepath extension; default: None.

            Currently supported filetypes are:
                - "json" to read the benchmark from a json file
                - "pickle" to read the benchmark from a pickle file
                - "dill" to read the benchmark from a pickle file using the "dill" library which supports function dumps

        :return:
            The benchmark instance.
        """
        filetype = filetype or os.path.splitext(filepath)[1].lstrip(".")
        match filetype:
            case "json":
                with open(filepath, "r") as file:
                    dump = json.load(fp=file)
            case "pickle":
                with open(filepath, "rb") as file:
                    dump = pickle.load(file=file)
            case "dill":
                with open(filepath, "rb") as file:
                    dump = dill.load(file=file)
            case other:
                raise AssertionError(f"Unknown filetype '{other}'")
        config = dump.pop("config")
        return cls(**dump, **config)

    @classmethod
    def describe(cls) -> str:
        """Describes the benchmark.

        :return:
            A string representing the textual description of the benchmark.
        """
        parameters = "\n  - ".join([str(p) for p in cls.PARAMETERS])
        return f"{cls.ALIAS.upper()}\n\n{cls.DESCRIPTION}\n\nPARAMETERS:\n  - {parameters}"

    def __init__(self, name: Optional[str] = None, seed: int = 42, **params):
        """
        :param name:
            The name of the benchmark, or None to use the benchmark alias; default: None.

        :param seed:
            The seed for random operations.

        :param params:
            A dictionary of parameters which must match the benchmark ones. If a parameter is not explicitly passed,
            its default value is used instead.
        """
        self.name: str = name or self.ALIAS
        self.seed: int = seed
        self.config: Dict[str, Any] = {}
        # validate given parameters or use default values for those not passed
        for p in self.PARAMETERS:
            name = p.name
            value = params.get(name) or p.default
            p.validate(value=value)
            self.config[name] = value
        # check that no additional parameters have been passed
        for p in params.keys():
            assert p in self.config, f"Parameter '{p}' is not a valid parameter for benchmark '{self.name}'"
        # build dataframe
        self.data: pd.DataFrame = self.generate()

    def __repr__(self) -> str:
        desc = f"{self.name} (seed: {self.seed})\n  - "
        desc += "\n  - ".join([f"{n}: {to_string(p)}" for n, p in self.config.items()])
        return desc

    def generate(self, **kwargs) -> pd.DataFrame:
        """Generates the benchmark data given a config that matches the benchmark parameters.

        :param kwargs:
            Any benchmark-specific argument.

        :return:
            A Dataframe containing the generated benchmark data.
        """
        raise NotImplementedError("Please implement abstract method 'generate_data'")

    def plot(self, **kwargs):
        """Plots the benchmark data.

        :param kwargs:
            Any benchmark-specific argument.
        """
        pass

    def serialize(self, filepath: str, filetype: Optional[str] = None):
        """Dumps the benchmark configuration into a file.

        :param filepath:
            The path of the file.

        :param filetype:
            The type of encoding, or None to infer it from the filepath extension; default: None.

            Currently supported filetypes are:
                - "txt" to dump the string representation of the benchmark
                - "json" to dump the benchmark as a json file
                - "pickle" to dump the benchmark as a pickle file
                - "dill" to dump the benchmark as a pickle file using the "dill" library which supports function dumps
        """
        dump = {"name": self.name, "seed": self.seed, "config": self.config}
        filetype = filetype or os.path.splitext(filepath)[1].lstrip(".")
        match filetype:
            case "txt":
                with open(filepath, "w") as file:
                    file.write(str(self))
            case "json":
                with open(filepath, "w") as file:
                    json.dump(dump, fp=file, indent=4)
            case "pickle":
                with open(filepath, "wb") as file:
                    pickle.dump(dump, file=file)
            case "dill":
                with open(filepath, "wb") as file:
                    dill.dump(dump, file=file)
            case other:
                raise AssertionError(f"Unknown filetype '{other}'")
