########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: test_basics.py 4463 2010-06-04 15:17:09Z faltet $
#
########################################################################

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
from carray.carrayExtension import chunk
import unittest


class chunkTest(unittest.TestCase):
    
    def test00(self):
        """Testing `toarray()` method"""
        a = np.arange(1e1)
        #a = np.linspace(-1, 1, 1e4)
        b = chunk(a)
        #print "b->", `b`
        assert_array_equal(a, b.toarray(), "Arrays are not equal")

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e1)
        b = chunk(a)
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e1)
        b = chunk(a)
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e1)
        b = chunk(a)
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = chunk(a)
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")


class carrayTest(unittest.TestCase):
    
    def test00(self):
        """Testing `toarray()` method"""
        a = np.arange(1e1)
        #a = np.linspace(-1, 1, 1e2)
        b = ca.carray(a, chunksize=10)
        #print "b->", `b`
        assert_array_equal(a, b.toarray(), "Arrays are not equal")

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = ca.carray(a, chunksize=1000)
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with no start"""
        a = np.arange(1e4)
        b = ca.carray(a, chunksize=1000)
        #print "b[:8000]->", `b[:8000]`
        assert_array_equal(a[:8000], b[:8000], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with no stop"""
        a = np.arange(1e4)
        b = ca.carray(a, chunksize=1000)
        #print "b[8000:]->", `b[8000:]`
        assert_array_equal(a[8000:], b[8000:], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with no start and no stop"""
        a = np.arange(1e4)
        b = ca.carray(a, chunksize=1000)
        #print "b[:]->", `b[::2]`
        assert_array_equal(a[::2], b[::2], "Arrays are not equal")

    def test05(self):
        """Testing `append()` method"""
        a = np.arange(1e1)
        b = ca.carray(a)
        b.append(a)
        #print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b.toarray(), "Arrays are not equal")

    def test06(self):
        """Testing `append()` method (small chunksize)"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=10)
        b.append(a)
        #print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b.toarray(), "Arrays are not equal")

    def test07(self):
        """Testing `append()` method (large append)"""
        a = np.arange(1e4)
        c = np.arange(2e5)
        b = ca.carray(a)
        b.append(c)
        #print "b->", `b`
        d = np.concatenate((a, c))
        assert_array_equal(d, b.toarray(), "Arrays are not equal")



def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(chunkTest))
    theSuite.addTest(unittest.makeSuite(carrayTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
