import unittest

from model import Structure


class TestStructure(unittest.TestCase):
    def test_names(self):
        structure = Structure(alias='alias', description='desc')
        structure.add_custom_variable(name='var')
        structure.add_custom_parameter(name='par', default=None)
        structure.add_generic_constraint(name='cst', check=lambda: True)
        structure.add_sample_metric(name='mtr', function=lambda sample: 0)
        # check variables
        structure.add_custom_variable(name='cst')
        with self.assertRaises(AssertionError, msg='Name clash between inputs should raise an exception'):
            structure.add_custom_variable(name='var')
        with self.assertRaises(AssertionError, msg='Name clash between inputs should raise an exception'):
            structure.add_custom_variable(name='par')
        # check parameters
        structure.add_custom_parameter(name='mtr', default=None)
        with self.assertRaises(AssertionError, msg='Name clash between inputs should raise an exception'):
            structure.add_custom_parameter(name='var', default=None)
        with self.assertRaises(AssertionError, msg='Name clash between inputs should raise an exception'):
            structure.add_custom_parameter(name='par', default=None)
        # check constraints
        structure.add_generic_constraint(name='var', check=lambda: True)
        structure.add_generic_constraint(name='par', check=lambda: True)
        structure.add_generic_constraint(name='mtr', check=lambda: True)
        with self.assertRaises(AssertionError, msg='Name clash between constraints should raise an exception'):
            structure.add_generic_constraint(name='cst', check=lambda: True)
        # check metrics
        structure.add_sample_metric(name='var', function=lambda sample: 0)
        structure.add_sample_metric(name='par', function=lambda sample: 0)
        structure.add_sample_metric(name='cst', function=lambda sample: 0)
        with self.assertRaises(AssertionError, msg='Name clash between metrics should raise an exception'):
            structure.add_sample_metric(name='mtr', function=lambda sample: 0)
