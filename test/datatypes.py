import unittest

import numpy as np

from model.datatypes import Sample, DataType, Variable, Parameter, Constraint, Metric


class TestDataTypes(unittest.TestCase):
    num: int = 10

    def test_samples(self):
        # test all types of inputs/output access
        inputs, output = {'x1': 1, 'x2': 2}, 3
        sample = Sample(inputs=inputs, output=output)
        self.assertEqual(inputs, sample.inputs, msg='Wrong inputs stored')
        self.assertEqual(inputs, sample['inputs'], msg='Wrong inputs stored')
        self.assertEqual(inputs, sample[0], msg='Wrong inputs stored')
        self.assertEqual(output, sample.output, msg='Wrong output stored')
        self.assertEqual(output, sample['output'], msg='Wrong output stored')
        self.assertEqual(output, sample[1], msg='Wrong output stored')
        self.assertEqual(str(sample), "Sample(inputs={'x1': 1, 'x2': 2}, output=3)", msg='Wrong string representation')

    def test_datatype(self):
        # test datatype without description
        dt = DataType(name='dt', description=None)
        self.assertEqual(dt.name, 'dt', msg='Wrong name stored')
        self.assertEqual(dt.description, None, msg='Wrong description stored')
        self.assertEqual(str(dt), 'DataType(dt)', msg='Wrong string representation')
        # test datatype with description
        dt = DataType(name='dt', description='desc')
        self.assertEqual(dt.name, 'dt', msg='Wrong name stored')
        self.assertEqual(dt.description, 'desc', msg='Wrong description stored')
        self.assertEqual(str(dt), 'DataType(dt) -> desc', msg='Wrong string representation')
        # test datatypes with wrong name
        with self.assertRaises(AssertionError, msg='Wrong name should raise an error'):
            DataType(name='wrong dt', description=None)
        with self.assertRaises(AssertionError, msg='Wrong name should raise an error'):
            DataType(name='0dt', description=None)
        with self.assertRaises(AssertionError, msg='Wrong name should raise an error'):
            DataType(name='dt?', description=None)

    def test_variable(self):
        rng = np.random.default_rng(0)
        # test variable info
        var = Variable(name='var', description='desc', dtype=str, domain='a*', check_domain=lambda v: v.startswith('a'))
        self.assertEqual(var.name, 'var', msg='Wrong name stored')
        self.assertEqual(var.description, 'desc', msg='Wrong description stored')
        self.assertEqual(var.domain, 'a*', msg='Wrong domain stored')
        self.assertEqual(var.dtype, str, msg='Wrong dtype stored')
        self.assertEqual(str(var), 'Variable(var) -> desc', msg='Wrong string representation')
        # test domain check
        for i in range(self.num):
            value = ''.join(rng.choice(['a', 'b', 'c'], size=3))
            self.assertEqual(var.check_domain(value), value.startswith('a'), msg='Domain function not working properly')
        # test dtype check
        self.assertTrue(var.check_dtype('string'), msg='Dtype function not working properly')
        self.assertFalse(var.check_dtype(1.0), msg='Dtype function not working properly')
        self.assertFalse(var.check_dtype(1), msg='Dtype function not working properly')
        # test float dtype supporting integers
        var = Variable(name='var', description='desc', dtype=float, domain='x > 0', check_domain=lambda v: v > 0)
        self.assertTrue(var.check_dtype(1), msg='Dtype function should support integers when dtype is float')

    def test_parameter(self):
        rng = np.random.default_rng(0)
        # test parameter info
        fn = lambda v: v.startswith('a')
        par = Parameter(name='par', default='abc', description='desc', dtype=str, domain='a*', check_domain=fn)
        self.assertEqual(par.name, 'par', msg='Wrong name stored')
        self.assertEqual(par.default, 'abc', msg='Wrong default value stored')
        self.assertEqual(par.description, 'desc', msg='Wrong description stored')
        self.assertEqual(par.domain, 'a*', msg='Wrong domain stored')
        self.assertEqual(par.dtype, str, msg='Wrong dtype stored')
        self.assertEqual(str(par), 'Parameter(par) -> desc', msg='Wrong string representation')
        # test domain check
        for i in range(self.num):
            value = ''.join(rng.choice(['a', 'b', 'c'], size=3))
            self.assertEqual(par.check_domain(value), value.startswith('a'), msg='Domain function not working properly')
        # test dtype check
        self.assertTrue(par.check_dtype('a'), msg='Dtype function not working properly')
        self.assertFalse(par.check_dtype(1.0), msg='Dtype function not working properly')
        self.assertFalse(par.check_dtype(1), msg='Dtype function not working properly')
        # test float dtype supporting integers
        fn = lambda v: v > 0
        par = Parameter(name='par', default=1.0, description='desc', dtype=float, domain='x > 0', check_domain=fn)
        self.assertTrue(par.check_dtype(1), msg='Dtype function should support integers when dtype is float')
        # test parameters with wrong default values
        with self.assertRaises(AssertionError, msg='Wrong default parameter should raise an error'):
            Parameter(name='p', default=0.1, description=None, dtype=int, domain='all', check_domain=lambda v: True)
        with self.assertRaises(AssertionError, msg='Wrong default parameter should raise an error'):
            Parameter(name='p', default=0.1, description=None, dtype=float, domain='all', check_domain=lambda v: v < 0)
        with self.assertRaises(AssertionError, msg='Wrong default parameter should raise an error'):
            Parameter(name='p', default=0.1, description=None, dtype=int, domain='all', check_domain=lambda v: v < 0)

    def test_constraint(self):
        rng = np.random.default_rng(0)
        # test constraint info
        cst = Constraint(name='cst', description='desc', check=lambda a, b, c: a * b > c)
        self.assertEqual(cst.name, 'cst', msg='Wrong name stored')
        self.assertEqual(cst.description, 'desc', msg='Wrong description stored')
        self.assertListEqual(cst.inputs, ['a', 'b', 'c'], msg='Wrong inputs stored')
        self.assertEqual(str(cst), 'Constraint(cst) -> desc', msg='Wrong string representation')
        # test constraint satisfaction check
        for i in range(self.num):
            values = rng.integers(-10, 10, size=3)
            satisfied = values[0] * values[1] > values[2]
            self.assertEqual(cst.check(*values), satisfied, msg='Constraint check function not working properly')
        # test constraint satisfaction signature
        self.assertTrue(cst.check(a=1, b=2, c=1), msg='Constraint check function not working properly')
        with self.assertRaises(TypeError, msg='Constraint check function has a wrong signature'):
            cst.check()
        with self.assertRaises(TypeError, msg='Constraint check function has a wrong signature'):
            cst.check(1)
        with self.assertRaises(TypeError, msg='Constraint check function has a wrong signature'):
            cst.check(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            cst.check(a=1, b=2, d=1)

    def test_metric(self):
        s1 = Sample(inputs={'a': -1, 'b': -2}, output=3)
        s2 = Sample(inputs={'a': 'a', 'b': 'b'}, output='c')
        # test output metric behaviour
        mtr = Metric(name='mtr', description='desc', evaluate=lambda s: s.output)
        self.assertEqual(mtr.name, 'mtr', msg='Wrong name stored')
        self.assertEqual(mtr.description, 'desc', msg='Wrong description stored')
        self.assertEqual(mtr.evaluate(s1), 3, msg='Metric evaluation not working properly')
        self.assertEqual(mtr.evaluate(s2), 'c', msg='Metric evaluation not working properly')
        # test input metric behaviour
        mtr = Metric(name='mtr', description='desc', evaluate=lambda s: s.inputs['a'] + s.inputs['b'])
        self.assertEqual(mtr.name, 'mtr', msg='Wrong name stored')
        self.assertEqual(mtr.description, 'desc', msg='Wrong description stored')
        self.assertEqual(mtr.evaluate(s1), -3, msg='Metric evaluation not working properly')
        self.assertEqual(mtr.evaluate(s2), 'ab', msg='Metric evaluation not working properly')
