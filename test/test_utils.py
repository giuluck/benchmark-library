import unittest

import numpy as np

from model.utils import stringify


class TestUtils(unittest.TestCase):
    def test_stringify(self):
        def dummy(x):
            return x

        # test None
        self.assertEqual(stringify(None), '', msg='Wrong representation of None')
        # test builtins
        self.assertEqual(stringify(1), '1', msg='Wrong representation of string')
        self.assertEqual(stringify(1.0), '1.0', msg='Wrong representation of string')
        self.assertEqual(stringify('1'), "'1'", msg='Wrong representation of string')
        # test types
        self.assertEqual(stringify(type(1)), 'int', msg='Wrong representation of type int')
        self.assertEqual(stringify(type(1.0)), 'float', msg='Wrong representation of type float')
        self.assertEqual(stringify(type('1')), 'str', msg='Wrong representation of type string')
        self.assertEqual(stringify(type({1})), 'set', msg='Wrong representation of type set')
        self.assertEqual(stringify(type([1])), 'list', msg='Wrong representation of type list')
        self.assertEqual(stringify(type((1,))), 'tuple', msg='Wrong representation of type tuple')
        self.assertEqual(stringify(type({1: 1})), 'dict', msg='Wrong representation of type dictionary')
        self.assertEqual(stringify(type(np.empty(1))), 'ndarray', msg='Wrong representation of type ndarray')
        self.assertEqual(stringify(type(self.test_stringify)), 'method', msg='Wrong representation of type method')
        self.assertEqual(stringify(type(dummy)), 'function', msg='Wrong representation of type function')
        # test iterables
        obj = {0: '0', (0, 1): [0, 1]}
        self.assertEqual(stringify(set(obj.keys())), '{0, (0, 1)}', msg='Wrong representation of set')
        self.assertEqual(stringify(list(obj.keys())), '[0, (0, 1)]', msg='Wrong representation of list')
        self.assertEqual(stringify(tuple(obj.keys())), '(0, (0, 1))', msg='Wrong representation of tuple')
        self.assertEqual(stringify(obj), "{0: '0', (0, 1): [0, 1]}", msg='Wrong representation of dictionary')
        # test callable
        self.assertEqual(stringify(dummy), 'dummy(x)', msg='Wrong representation of function')
        self.assertEqual(stringify(lambda x: x), '<lambda>(x)', msg='Wrong representation of lambda')
        self.assertEqual(stringify(self.test_stringify), 'test_stringify()', msg='Wrong representation of method')
        # test prefix/suffix
        self.assertEqual(stringify(None, prefix='p', suffix='s'), '', msg='Wrong prefix/suffix for None')
        self.assertEqual(stringify(1, prefix='p', suffix='s'), 'p1s', msg='Wrong prefix/suffix for int')
        self.assertEqual(stringify(1.0, prefix='p', suffix='s'), 'p1.0s', msg='Wrong prefix/suffix for float')
        self.assertEqual(stringify('v', prefix='p', suffix='s'), "p'v's", msg='Wrong prefix/suffix for string')
