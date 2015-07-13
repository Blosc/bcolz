# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: January 2, 2014
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

"""
Unit tests related to the handling of arrays of objects
-------------------------------------------------------

Notes on object handling:

1. Only one dimensional arrays of objects are handled

2. Composite dtypes that contains objects are currently not handled.

"""

import unittest
from unittest import TestCase

import numpy as np
import bcolz
from bcolz.tests.common import MayBeDiskTest


class ObjectCarrayTest(MayBeDiskTest):
    def test_carray_1d_source(self):
        """Testing carray of objects, 1d source"""
        src_data = ['s'*i for i in range(10)]
        carr = bcolz.carray(src_data, dtype=np.dtype('O'))

        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i], src_data[i])
            self.assertEqual(carr[i], src_data[i])

    def test_carray_2d_source(self):
        """Testing carray of objects, 2d source

        Expected result will be a 1d carray whose elements are
        containers holding the inner dimension
        """
        src_data = [(i, 's'*i) for i in range(10)]
        carr = bcolz.carray(src_data, dtype=np.dtype('O'))
        # note that carray should always create a 1 dimensional
        # array of objects.
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_carray_tuple_source(self):
        """Testing a carray of objects that are tuples

        This uses a numpy container as source. Tuples should be
        preserved
        """
        src_data = np.empty((10,), dtype=np.dtype('O'))
        src_data[:] = [(i, 's'*i) for i in range(src_data.shape[0])]
        carr = bcolz.carray(src_data)
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        self.assertEqual(type(carr[0]), tuple)
        self.assertEqual(type(carr[0]), type(src_data[0]))
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_carray_record(self):
        """Testing carray handling of record dtypes containing
        objects.  They must raise a type error exception, as they are
        not supported
        """
        src_data = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, bcolz.carray,
                          src_data, dtype=np.dtype('O,O'))

    def test_carray_record_as_object(self):
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        carr = bcolz.carray(src_data, dtype=np.dtype('O'))
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    # The following tests document different alternatives in handling
    # input data which would infer a record dtype in the resulting
    # carray.
    #
    # option 1: fail with a type error as if the dtype was
    # explicit
    #
    # option 2: handle it as an array of arrays of objects.
    def test_carray_record_inferred_opt1(self):
        """Testing carray handling of inferred record dtypes
        containing objects.  When there is no explicit dtype in the
        carray constructor, the dtype is inferred. This test checks
        that an inferred dtype results in a type error.
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, bcolz.carray, src_data)

    # This test is disabled.  option 1 above has been implemented.
    def _test_carray_record_inferred_opt2(self):
        """Testing carray handling of inferred record dtypes
        containing objects.  When there is no explicit dtype in the
        carray constructor, the dtype becomes 'O', and the carrays
        behaves accordingly (one dimensional)
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]

        carr = bcolz.carray(src_data)
        # note: this is similar as if it was created with dtype='O'
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_create_unsafe_carray_with_unsafe_data(self):
        """ We introduce a safe keyword arg which removes dtype checking.
        We don't want this to interfere with creation.
        """
        b = bcolz.carray([1, 2, 3], dtype='i4', safe=False)
        self.assertEqual(b.safe, False)
        self.assertEqual(b[0], 1)


class ObjectCarraymemoryTest(ObjectCarrayTest, TestCase):
    disk = False


class ObjectCarrayDiskTest(ObjectCarrayTest, TestCase):
    disk = True


if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# coding: utf-8
# fill-column: 78
# End:
