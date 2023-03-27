# Interactive Benchmarks

This library allows to both **build** and, accordingly, **use** a wide range of benchmark problems sharing the same structure.

The library is designed for two kinds of people:
* **programmers**, who can build a new benchmark (or a group of such) leveraging the internal API
* **final users**, who want to test their algorithms and models on the proposed benchmarks

## **How To Use A Benchmark**

The set of available benchmarks can be found in the `benchmarks` package.
This package consists of a set of subpackages, each of which containing:
* a `__init__.py` file with the source code implementing a single benchmark or a set of semantically correlated ones
* a `README.md` file with a detailed description of the benchmarks (i.e., their mathematical formulations, their input variables and parameters, etc.)
* a `requirements.txt` file with the list of benchmark-specific python requirements

Each benchmark can be accessed by importing its respective subpackage.
Let us consider the case of the `Ackley` benchmark, which can be found in the `blackbox` subpackage.
You can create an instance of this benchmark in this way:
```python
from benchmarks import blackbox as bb
benchmark = bb.Ackley()
```

### _1. Benchmark Structure_

Each benchmark share a common structure, which is made of:
* a set of `variables` representing the input space of the function to query
* a set of `parameters` representing the configuration of a specific benchmark instance
* a set of `constraints` which perform some sanity checks on the inputs (_variables + parameters_) which must be satisfied
* a set of `metrics` which are used to evaluate the goodness of a queried sample

All this information is stored as class-level properties along with a benchmark `alias` and a brief `description` of it.
It is also possible to print all this information from within the code by calling the `describe` class method.
Additionally, the `brief` parameter allows to include/exclude additional information in addition to their names and description.
```python
description = bb.Ackley.describe(brief=False)
print(description)
```

### _2. Benchmark Instance_

While the previous properties are defined at class-level since they are related to the benchmark definition, each benchmark object has a number of instance-level properties as well:
* a `name`, which can be different from the benchmark `alias` as it is bounded to the benchmark instance rather than its class
* an optional `seed` value, which is used to build an internal random number generator in case the benchmark uses random number operations
* a `configuration` dictionary, which contains the values of the benchmark parameters bounded to that specific benchmark instance
* a list of `samples`, which stores the history of queries as pairs of query inputs and its respective output

Custom values for the configuration can be passed in the benchmark constructor.
Their names should match those of the benchmark parameters, and the given values must respect the parameters domains.
E.g., the Ackley benchmark has four different parameters, namely the three real values `a`, `b`, and `c`, and the integer `dim` which represent the expected size of the input vector.
```python
instance = bb.Ackley(name='instance', a=20, dim=2)
```
> _Note: all the parameters must have a default value, which is used when they are not specified in the constructor._

Additionally, each benchmark object can be serialized and deserialized using the `serialize` and the `load` methods.
```python
instance.serialize(filepath='ackley_instance.dill')
loaded = Benchmark.load(filepath='ackley_instance.dill')
```
The loaded instance will have the same name, seed, and configuration as the original one.
Moreover, if the instance was queried before the serialization, the list of samples will be identically populated, and the random number generator will be in the same state as where it was left.

### _3. Querying and Evaluating_

Once a benchmark object is created, you can query it using the `query` method.
This method takes a series of input values whose names must match that of the benchmark variables.
E.g., the Ackley function has a single variable `x`, which must be a list/vector of real values:
```python
output = instance.query(x=[1.0, 2.0])
```

The output of the query method can be of any type, depending on the benchmark.
In this case, it is a floating point value representing the evaluation of the 2D Ackley function in the point $[1., 2.]$.
When the query method is called, it performs consistency checks on the variables domain and on the global constraints.
E.g., in the case of the Ackley benchmark, the only global constraint checks that the input vector has the correct size.

Moreover, when the query method is called, it automatically appends a `Sample` object in the `samples` list.
`Sample` objects contain two fields, i.e., the given input variables stored in a dictionary, and the respective output computed by the query function.
Each sample can be evaluated as for the benchmark metrics using the `evaluate` method:
```python
sample = instance.samples[0]
instance.evaluate(sample=sample)
```
This method returns a dictionary of metric values indexed by their respective name.
When no explicit sample is passed as input, the last queried sample is used for the evaluation.

Finally, it is possible to return the whole querying history using the `history` method:
```python
history = instance.history(plot=False)
```
The returned history object is a pandas dataframe where each column represent a metric and each row a sample in the list.
It is also possible to automatically plot each metric using the `plot` parameter.

## **How To Build A New Benchmark**

Along with the possibility to use the benchmarks, this library provides an internal API to define new use cases.
The first thing to do when implementing a new benchmark (or a set of such) is to **create a new subpackage** within the `benchmarks` one.

This subpackage should contain:
* a `README.md` file with a detailed description of the benchmarks (i.e., their mathematical formulations, their input variables and parameters, etc.)
* a `requirements.txt` file with the list of benchmark-specific python requirements
* a `__init__.py` file with the source code implementing a single benchmark or a set of semantically correlated ones

Finally, it is good practice to reference the subpackage in the `__init__.py` file of the `benchmarks` package:
```python
from benchmarks import blackbox
from benchmarks import epidemics
# add your new subpackage here
```

### _1. Benchmark Structure_

Each benchmark should extend the `Benchmark` base class, which contains two abstract methods.
The first of them is the static method `build`, which allows to define the properties of the benchmark.

As an example, let us see the implementation of that method for the `Ackley` benchmark:
```python
@staticmethod
from model import Structure

def build(structure: Structure):
    # variables
    structure.add_custom_variable('x', dtype=list, description="the input vector")
    # parameters
    structure.add_numeric_parameter('a', default=20, description="the Ackley function 'a' parameter")
    structure.add_numeric_parameter('b', default=0.2, description="the Ackley function 'b' parameter")
    structure.add_numeric_parameter('c', default=2 * np.pi, description="the Ackley function 'c' parameter")
    structure.add_numeric_parameter('dim', default=1, integer=True, lb=1, description="the input vector dimension")
    # constraints
    structure.add_generic_constraint(
        name='input_dim',
        check=lambda x, dim: len(x) == dim,
        description="the input vector should have the correct input dimension 'dim'"
    )
    # metrics
    structure.add_reference_metric(
        name='gap',
        metric='mae',
        reference=0.0,
        description='absolute gap from the optimum'
    )
```

We can see that this benchmark contains:
* one input variable (`x`), which must be a list
* three real-valued parameters (`a`, `b`, and `c`), and an integer one (`dim`), each with their respective default value
* a global constraint checking that the input vector has the correct size
* a metric which measures the absolute gap from the (known) optimum value
> _Note: all the names must be valid identifiers, i.e., they must start with a letter and contain no special characters apart from the underscore._
> _Also, there must be no name clash between metrics, constraints, and inputs (variables + parameters)._

All this information will be used both to perform the required checks on the given input values and to provide the benchmark description.
Along with this, the `Structure` object also contain two additional fields, `alias` and `description` representing, respectively, the benchmark alias and a textual description of it.
By default, the `Structure` object comes with predefined alias and description which are retrieved from the benchmark class name and its docstring, respectively.
However, it is possible to change their default values using:
```python
structure.alias = 'alias'
structure.description = 'description'
```

> _Note: the `Strucure` object provides different utility methods to include variables, parameters, constraints, and metrics with certain domains and types._

### _2. The Query Method_

The second abstract method which must be implemented is the `query` one.
This method contains the code to generate the benchmark output given a set of inputs which must match the benchmark variables.
For the `Ackley` benchmark, this method is implemented as follows:
```python
import numpy as np
from model import querymethod

@querymethod
def query(self, x: list) -> float:
    x = np.array(x)
    term1 = self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2) / self.dim))
    term2 = np.exp(np.sum(np.cos(self.c * x)) / self.dim)
    return term1 + term2 - self.a - np.e
```

Notice that only the variable is passed as inputs, while the parameter configuration is retrieved from the `self` object (_see the next subsection for a more detailed explanation_).

Moreover, this method must be decorated with the `querymethod` decorator.
This decorator takes care of performing the necessary sanity checks on the input variables domains and global constraints, as well as storing the results in the `samples` list.

When implementing the query method, please remind to:
* include all the python packages you used in the `requirements.txt` file
* use the internal `self._rng` random number generator to perform random number operations, in order to have replicable results

### _3. The Init Method_

In principle, the default `__init__` method of the `Benchmark` class is enough to make your benchmark work.
However, for the end users it is more beneficial to see the real names of the parameters rather than a generic `**configuration` argument in the signature, hence it is good practice to define your own constructor.
The signature constructor of a benchmark should always contain:
* the `name` parameter, i.e., an optional string
* the `seed` parameter if and only if the benchmark expects some random number operations
* the explicit list of configuration parameters with the correct default values and types

For example, here is the `__init__` method of the `Ackley` benchmark:
```python
def __init__(self, name: Optional[str] = None, a: float = 20, b: float = 0.2, c: float = 2 * np.pi, dim: int = 1):
    super(Ackley, self).__init__(name=name, seed=None, a=a, b=b, c=c, dim=dim)
```
> _Note: the `seed` parameter is not required since the function is deterministic.`
 
Finally, it is beneficial to either add a property for each parameter value in the constructor.
This will allow for direct access to the configuration from the `self` object as done within the query method showed in the previous section.

Since the parameters are automatically stored in the internal `_configuration` field of the benchmark, it is better to define instance properties rather than new fields so to avoid inconsistencies between the fields values and their values in the configuration dictionary.
Taking the `Ackley` benchmark as an example, each the respective property of configuration parameter could be defined in this way:
```python
@property
def a(self) -> float:
    return self._configuration['a']
```
