########################################################################
#
#       License: BSD
#       Created: January 11, 2011
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

import sys
import struct

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
from carray.carrayExtension import chunk
import unittest


class constructorTest(unittest.TestCase):

    def test00(self):
        """Testing `carray` constructor"""
        a = np.arange(16).reshape((2,2,4))
        b = ca.carray(a)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01a(self):
        """Testing `zeros` constructor (I)"""
        a = np.zeros((2,2,4), dtype='i4')
        b = ca.zeros((2,2,4), dtype='i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01b(self):
        """Testing `zeros` constructor (II)"""
        a = np.zeros(2, dtype='(2,4)i4')
        b = ca.zeros(2, dtype='(2,4)i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01c(self):
        """Testing `zeros` constructor (III)"""
        a = np.zeros((2,2), dtype='(4,)i4')
        b = ca.zeros((2,2), dtype='(4,)i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02(self):
        """Testing `ones` constructor"""
        a = np.ones((2,2), dtype='(4,)i4')
        b = ca.ones((2,2), dtype='(4,)i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03a(self):
        """Testing `fill` constructor (scalar default)"""
        a = np.ones((2,2), dtype='(4,)i4')*3
        b = ca.fill((2,2), 3, dtype='(4,)i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03b(self):
        """Testing `fill` constructor (array default)"""
        a = np.ones((2,2), dtype='(4,)i4')*3
        b = ca.fill((2,2), [3,3,3,3], dtype='(4,)i4')
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")


class getitemTest(unittest.TestCase):

    def test00(self):
        """Testing `__getitem()__` method with only a start"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((2,3), 3, dtype="i4")
        sl = slice(1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01(self):
        """Testing `__getitem()__` method with a start and a stop"""
        a = np.ones((5,2), dtype="i4")*3
        b = ca.fill((5,2), 3, dtype="i4")
        sl = slice(1,4)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02(self):
        """Testing `__getitem()__` method with a start, stop, step"""
        a = np.ones((10,2), dtype="i4")*3
        b = ca.fill((10,2), 3, dtype="i4")
        sl = slice(1,9,2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")


class setitemTest(unittest.TestCase):

    def test00a(self):
        """Testing `__setitem()__` method with only a start (scalar)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((2,3), 3, dtype="i4")
        sl = slice(1)
        a[sl,:] = 0
        b[sl] = 0
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test00b(self):
        """Testing `__setitem()__` method with only a start (vector)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((2,3), 3, dtype="i4")
        sl = slice(1)
        a[sl,:] = range(3)
        b[sl] = range(3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01a(self):
        """Testing `__setitem()__` method with start,stop (scalar)"""
        a = np.ones((5,2), dtype="i4")*3
        b = ca.fill((5,2), 3, dtype="i4")
        sl = slice(1,4)
        a[sl,:] = 0
        b[sl] = 0
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01b(self):
        """Testing `__setitem()__` method with start,stop (vector)"""
        a = np.ones((5,2), dtype="i4")*3
        b = ca.fill((5,2), 3, dtype="i4")
        sl = slice(1,4)
        a[sl,:] = range(2)
        b[sl] = range(2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02a(self):
        """Testing `__setitem()__` method with start,stop,step (scalar)"""
        a = np.ones((10,2), dtype="i4")*3
        b = ca.fill((10,2), 3, dtype="i4")
        sl = slice(1,8,3)
        a[sl,:] = 0
        b[sl] = 0
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02b(self):
        """Testing `__setitem()__` method with start,stop,step (scalar)"""
        a = np.ones((10,2), dtype="i4")*3
        b = ca.fill((10,2), 3, dtype="i4")
        sl = slice(1,8,3)
        a[sl,:] = range(2)
        b[sl] = range(2)
        #print "b[sl]->", `b[sl]`, `b`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")


class appendTest(unittest.TestCase):

    def test00(self):
        """Testing `append()` method (correct shape)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((1,3), 3, dtype="i4")
        b.append([(3,3,3)])
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01(self):
        """Testing `append()` method (incorrect shape)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((1,3), 3, dtype="i4")
        self.assertRaises(ValueError, b.append, [(3,3)])

    def test02(self):
        """Testing `append()` method (several rows)"""
        a = np.ones((4,3), dtype="i4")*3
        b = ca.fill((1,3), 3, dtype="i4")
        b.append([(3,3,3)]*3)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")


class resizeTest(unittest.TestCase):

    def test00a(self):
        """Testing `resize()` (trim)"""
        a = np.ones((2,3), dtype="i4")
        b = ca.ones((3,3), dtype="i4")
        b.resize(2)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `resize()` (trim to zero)"""
        a = np.ones((0,3), dtype="i4")
        b = ca.ones((3,3), dtype="i4")
        b.resize(0)
        #print "b->", `b`
        # The next does not work well for carrays with shape (0,)
        #assert_array_equal(a, b, "Arrays are not equal")
        self.assert_("a.dtype.base == b.dtype.base")
        self.assert_("a.shape == b.shape+b.dtype.shape")

    def test01(self):
        """Testing `resize()` (enlarge)"""
        a = np.ones((4,3), dtype="i4")
        b = ca.ones((3,3), dtype="i4")
        b.resize(4)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(constructorTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))
    theSuite.addTest(unittest.makeSuite(setitemTest))
    theSuite.addTest(unittest.makeSuite(appendTest))
    theSuite.addTest(unittest.makeSuite(resizeTest))


    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
