import os
import unittest
from typing import Optional

import numpy as np

from model import Benchmark, Structure, querymethod
from model.datatypes import Sample


class Dummy(Benchmark):
    """The description."""

    @staticmethod
    def build(structure: Structure):
        structure.add_positive_variable('var', description='the variable')
        structure.add_negative_parameter('par', default=-2, integer=True, strict=True, description='the parameter')
        structure.add_generic_constraint('cst', check=lambda var, par: var >= -par, description='the constraint')
        structure.add_sample_metric('mtr', function=lambda sample: sample.output, description='the metric')

    @querymethod
    def query(self, var: float) -> float:
        return var ** (-self._configuration['par'])

    def __init__(self, name: Optional[str] = None, seed: int = 42, par: int = -2):
        super(Dummy, self).__init__(name=name, seed=seed, par=par)

    def noise(self, size: int) -> np.ndarray:
        return self._rng.normal(size=size)

    brief_description: str = """
    DUMMY

    The description.
    
    Variables:
      * var: the variable
    
    Parameters:
      * par: the parameter
    
    Constraints:
      * cst: the constraint

    Metrics:
      * mtr: the metric
    """.replace('\n    ', '\n').strip('\n')

    full_description: str = """
    DUMMY
    
    The description.
    
    Variables:
      * var: the variable
        - domain: [0, +inf[
        -   type: float
    
    Parameters:
      * par: the parameter
        - default: -2
        -  domain: ]-inf, 0[
        -    type: int
    
    Constraints:
      * cst: the constraint
    
    Metrics:
      * mtr: the metric
    """.replace('\n    ', '\n').strip('\n')


class TestBenchmark(unittest.TestCase):
    size: int = 100

    def test_class(self):
        # test class properties
        self.assertEqual(Dummy.alias, 'Dummy', msg='Wrong benchmark alias')
        self.assertEqual(Dummy.description, 'The description.', msg='Wrong benchmark description')
        self.assertEqual(Dummy.variables, dict(var='the variable'), msg='Wrong benchmark variables')
        self.assertEqual(Dummy.parameters, dict(par='the parameter'), msg='Wrong benchmark parameters')
        self.assertEqual(Dummy.constraints, dict(cst='the constraint'), msg='Wrong benchmark constraints')
        self.assertEqual(Dummy.metrics, dict(mtr='the metric'), msg='Wrong benchmark metrics')
        # test class description
        self.assertEqual(Dummy.describe(brief=True), Dummy.brief_description, msg='Wrong brief description')
        self.assertEqual(Dummy.describe(brief=False), Dummy.full_description, msg='Wrong full description')

    # noinspection PyTypeChecker
    def test_instance(self):
        # test default instance properties
        benchmark = Dummy()
        self.assertEqual(benchmark.seed, 42, msg='Wrong benchmark seed')
        self.assertEqual(benchmark.name, 'dummy', msg='Wrong benchmark name')
        self.assertListEqual(benchmark.samples, [], msg='Wrong benchmark samples')
        self.assertDictEqual(benchmark.configuration, dict(par=-2), msg='Wrong benchmark configuration')
        benchmark = Dummy(name='name', seed=0, par=-3)
        # test custom instance properties
        self.assertEqual(benchmark.seed, 0, msg='Wrong benchmark seed')
        self.assertEqual(benchmark.name, 'name', msg='Wrong benchmark name')
        self.assertListEqual(benchmark.samples, [], msg='Wrong benchmark samples')
        self.assertDictEqual(benchmark.configuration, dict(par=-3), msg='Wrong benchmark configuration')
        # test wrong instance properties
        with self.assertRaises(AssertionError, msg='Init should raise an error when given parameter is not in domain'):
            Dummy(par=0)
        with self.assertRaises(AssertionError, msg='Init should raise an error when given parameter is not in domain'):
            Dummy(par=1)
        with self.assertRaises(AssertionError, msg='Init should raise an error when given parameter is of wrong dtype'):
            Dummy(par=-1.0)

    def test_evaluation(self):
        benchmark = Dummy()
        # test wrong query and evaluation calls
        with self.assertRaises(AssertionError, msg='Evaluation should raise an error when no samples are available'):
            benchmark.evaluate()
        with self.assertRaises(AssertionError, msg='Query should raise an error when given input is not in domain'):
            benchmark.query(var=-1)
        with self.assertRaises(AssertionError, msg='Query should raise an error when constraint is not satisfied'):
            benchmark.query(var=1.0)
        # test correct query call
        output = benchmark.query(var=4)
        self.assertAlmostEqual(output, 16.0, msg='Wrong output returned from query method')
        samples = [Sample(inputs=dict(var=4), output=16.0)]
        self.assertListEqual(samples, benchmark.samples, msg='Wrong samples list after query call')
        output = benchmark.query(var=3)
        self.assertAlmostEqual(output, 9.0, msg='Wrong output returned from query method')
        samples += [Sample(inputs=dict(var=3), output=9.0)]
        self.assertListEqual(samples, benchmark.samples, msg='Wrong samples list after query call')
        # test correct evaluation call
        evaluation = dict(mtr=9.0)
        self.assertDictEqual(evaluation, benchmark.evaluate().to_dict(), msg='Wrong benchmark evaluation')
        self.assertDictEqual(evaluation, benchmark.evaluate(samples[1]).to_dict(), msg='Wrong benchmark evaluation')
        evaluation = dict(mtr=16.0)
        self.assertDictEqual(evaluation, benchmark.evaluate(samples[0]).to_dict(), msg='Wrong benchmark evaluation')

    def test_serialization(self):
        rng = np.random.default_rng(0)
        benchmark = Dummy(name='name', seed=0, par=-3)
        # test random number operations
        benchmark.query(var=3)
        benchmark_noise, rng_noise = list(benchmark.noise(size=self.size)), list(rng.normal(size=self.size))
        self.assertListEqual(benchmark_noise, rng_noise, msg='Wrong random operation')
        # test properties of loaded object
        benchmark.serialize('name.dill')
        benchmark = Benchmark.load('name.dill')
        self.assertEqual(benchmark.seed, 0, msg='Wrong loaded seed')
        self.assertEqual(benchmark.name, 'name', msg='Wrong loaded name')
        self.assertListEqual(benchmark.samples, [Sample(inputs=dict(var=3), output=27.0)], msg='Wrong loaded samples')
        self.assertDictEqual(benchmark.configuration, dict(par=-3), msg='Wrong loaded configuration')
        # test random number operations of loaded object
        benchmark_noise, rng_noise = list(benchmark.noise(size=self.size)), list(rng.normal(size=self.size))
        self.assertListEqual(benchmark_noise, rng_noise, msg='Wrong loaded random operation')
        os.remove('name.dill')
