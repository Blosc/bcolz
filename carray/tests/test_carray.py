########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: test_basics.py 4463 2010-06-04 15:17:09Z faltet $
#
########################################################################

import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
from carray.carrayExtension import chunk
import unittest


class chunkTest(unittest.TestCase):

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e1)
        b = chunk(a, cparams=ca.cparams())
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e1)
        b = chunk(a, cparams=ca.cparams())
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e1)
        b = chunk(a, cparams=ca.cparams())
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = chunk(a, cparams=ca.cparams())
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")


class carrayTest(unittest.TestCase):

    def test01(self):
        """Testing `__getitem()__` method with only a start"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test01b(self):
        """Testing `__getitem()__` method with only a (negative) start"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[-1]->", `b[-1]`
        self.assert_(a[-1] == b[-1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test02b(self):
        """Testing `__getitem()__` method with ranges (negative start)"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[-3:]->", `b[-3:]`
        assert_array_equal(a[-3:], b[-3:], "Arrays are not equal")

    def test02c(self):
        """Testing `__getitem()__` method with ranges (negative stop)"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1:-3]->", `b[1:-3]`
        assert_array_equal(a[1:-3], b[1:-3], "Arrays are not equal")

    def test02d(self):
        """Testing `__getitem()__` method with ranges (negative start, stop)"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[-3:-1]->", `b[-3:-1]`
        assert_array_equal(a[-3:-1], b[-3:-1], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with negative steps"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=100)
        #print "b[::-3]->", `b[::-3]`
        self.assertRaises(NotImplementedError, b.__getitem__,
                          slice(None, None,-3))

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
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test06(self):
        """Testing `append()` method (small chunksize)"""
        a = np.arange(1e1)
        b = ca.carray(a, chunksize=10)
        b.append(a)
        #print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test07(self):
        """Testing `append()` method (large append)"""
        a = np.arange(1e4)
        c = np.arange(2e5)
        b = ca.carray(a)
        b.append(c)
        #print "b->", `b`
        d = np.concatenate((a, c))
        assert_array_equal(d, b[:], "Arrays are not equal")

    def test08(self):
        """Testing __len__()"""
        a = np.arange(111)
        b = ca.carray(a)
        self.assert_(len(a) == len(b), "Arrays do not have the same length")

    def test09(self):
        """Testing __sizeof__() (big carrays)"""
        a = np.arange(2e5)
        b = ca.carray(a)
        #print "size b uncompressed-->", b.nbytes
        #print "size b compressed  -->", b.cbytes
        self.assert_(sys.getsizeof(b) < b.nbytes,
                     "carray does not seem to compress at all")

    def test10(self):
        """Testing __sizeof__() (small carrays)"""
        a = np.arange(111)
        b = ca.carray(a)
        #print "size b uncompressed-->", b.nbytes
        #print "size b compressed  -->", b.cbytes
        self.assert_(sys.getsizeof(b) > b.nbytes,
                     "carray compress too much??")


class copyTest(unittest.TestCase):

    def test00(self):
        """Testing copy() without params"""
        a = np.arange(111)
        b = ca.carray(a)
        c = b.copy()
        c.append(np.arange(111, 122))
        self.assert_(len(b) == 111, "copy() does not work well")
        self.assert_(len(c) == 122, "copy() does not work well")
        r = np.arange(122)
        assert_array_equal(c[:], r, "incorrect correct values after copy()")

    def test01(self):
        """Testing copy() with higher compression"""
        a = np.linspace(-1., 1., 1e4)
        b = ca.carray(a)
        c = b.copy(cparams=ca.cparams(clevel=9))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes > c.cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lesser compression"""
        a = np.linspace(-1., 1., 1e4)
        b = ca.carray(a)
        c = b.copy(cparams=ca.cparams(clevel=1))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes < c.cbytes, "clevel not changed")

    def test03(self):
        """Testing copy() with no shuffle"""
        a = np.linspace(-1., 1., 1e4)
        b = ca.carray(a)
        c = b.copy(cparams=ca.cparams(shuffle=False))
        #print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assert_(b.cbytes < c.cbytes, "shuffle not changed")


class IterTest(unittest.TestCase):

    def test00(self):
        """Testing `iter()` method"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter1->", sum(b)
        #print "sum iter2->", sum((v for v in b))
        self.assert_(sum(a) == sum(b), "Sums are not equal")
        self.assert_(sum((v for v in a)) == sum((v for v in b)),
                     "Sums are not equal")

    def test01a(self):
        """Testing `iter()` method with a positive start"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(3))
        self.assert_(sum(a[3:]) == sum(b.iter(3)), "Sums are not equal")

    def test01b(self):
        """Testing `iter()` method with a negative start"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(-3))
        self.assert_(sum(a[-3:]) == sum(b.iter(-3)), "Sums are not equal")

    def test02a(self):
        """Testing `iter()` method with positive start, stop"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(3, 24))
        self.assert_(sum(a[3:24]) == sum(b.iter(3, 24)), "Sums are not equal")

    def test02b(self):
        """Testing `iter()` method with negative start, stop"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(-24, -3))
        self.assert_(sum(a[-24:-3]) == sum(b.iter(-24, -3)),
                     "Sums are not equal")

    def test02c(self):
        """Testing `iter()` method with positive start, negative stop"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(24, -3))
        self.assert_(sum(a[24:-3]) == sum(b.iter(24, -3)),
                     "Sums are not equal")

    def test03a(self):
        """Testing `iter()` method with only step"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(step=4))
        self.assert_(sum(a[::4]) == sum(b.iter(step=4)),
                     "Sums are not equal")

    def test03b(self):
        """Testing `iter()` method with start, stop, step"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        #print "sum iter->", sum(b.iter(3, 24, 4))
        self.assert_(sum(a[3:24:4]) == sum(b.iter(3, 24, 4)),
                     "Sums are not equal")

    def test03c(self):
        """Testing `iter()` method with negative step"""
        a = np.arange(101)
        b = ca.carray(a, chunksize=9)
        self.assertRaises(NotImplementedError, b.iter, 0, 1, -3)


class whereTest(unittest.TestCase):

    def test00(self):
        """Testing `where()` iterator (all true values)"""
        a = np.arange(1, 11) > 0
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.where()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.where()]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test01(self):
        """Testing `where()` iterator (all false values)"""
        a = np.arange(1, 11) < 0
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.where()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.where()]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test03(self):
        """Testing `where()` iterator (mix of true/false values)"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.where()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.where()]
        self.assert_(wt == cwt, "where() does not work correctly")


class getifTest(unittest.TestCase):

    def test00(self):
        """Testing `getif()` iterator (all true values)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v>0]
        cwt = [v for v in b.getif(a>0)]
        #print "numpy ->", [v for v in a if v>0]
        #print "getif ->", [v for v in b.getif(a>0)]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test01(self):
        """Testing `getif()` iterator (all false values)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<0]
        cwt = [v for v in b.getif(a<0)]
        #print "numpy ->", [v for v in a if v<0]
        #print "getif ->", [v for v in b.getif(a<0)]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test02a(self):
        """Testing `getif()` iterator (mix of true/false values, I)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.getif(a<=5)]
        #print "numpy ->", [v for v in a if v<=5]
        #print "getif ->", [v for v in b.getif(a<=5)]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test02b(self):
        """Testing `getif()` iterator (mix of true/false values, II)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5 and v>2]
        cwt = [v for v in b.getif((a<=5) & (a>2))]
        #print "numpy ->", [v for v in a if v<=5 and v>2]
        #print "getif ->", [v for v in b.getif((a<=5) & (a>2))]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test02c(self):
        """Testing `getif()` iterator (mix of true/false values, III)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5 or v>8]
        cwt = [v for v in b.getif((a<=5) | (a>8))]
        #print "numpy ->", [v for v in a if v<=5 or v>8]
        #print "getif ->", [v for v in b.getif((a<=5) | (a>8))]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test03(self):
        """Testing `getif()` iterator (using a boolean carray)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.getif(ca.carray(a<=5))]
        #print "numpy ->", [v for v in a if v<=5]
        #print "getif ->", [v for v in b.getif(ca.carray(a<=5))]
        self.assert_(wt == cwt, "getif() does not work correctly")

    def test04a(self):
        """Testing `getif()` iterator (using bool in fancy indexing)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = a[a<5]
        cwt = b[a<5]
        #print "numpy ->", a[a<5]
        #print "getif ->", b[a<5]
        assert_array_equal(wt, cwt, "getif() does not work correctly")

    def test04b(self):
        """Testing `getif()` iterator (using carray bool in fancy indexing)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = a[(a<5)|(a>9)]
        cwt = b[ca.carray((a<5)|(a>9))]
        #print "numpy ->", a[(a<5)|(a>9)]
        #print "getif ->", b[ca.carray((a<5)|(a>9))]
        assert_array_equal(wt, cwt, "getif() does not work correctly")


class fancy_indexingTest(unittest.TestCase):

    def test00(self):
        """Testing fancy indexing (short list)"""
        a = np.arange(1,111)
        b = ca.carray(a)
        c = b[[3,1]]
        r = a[[3,1]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing (large list, numpy)"""
        a = np.arange(1,1e4)
        b = ca.carray(a)
        idx = np.random.randint(1000, size=1000)
        c = b[idx]
        r = a[idx]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing (empty list)"""
        a = np.arange(101)
        b = ca.carray(a)
        c = b[[]]
        r = a[[]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        a = np.arange(1,101)
        b = ca.carray(a)
        c = b[[1.1, 3.3]]
        r = a[[1.1, 3.3]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1,101)
        b = ca.carray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(KeyError, b.__getitem__, idx)


class fromiterTest(unittest.TestCase):

    def test00(self):
        """Testing fromiter (short iter)"""
        a = np.arange(1,111)
        b = ca.fromiter(iter(a), dtype='i4')
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test01(self):
        """Testing fromiter (long iter)"""
        a = np.arange(1e4)
        #b = ca.fromiter(iter(a), dtype='f8', count=int(1e4))
        b = ca.fromiter(iter(a), dtype='f8', count=-1)
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test02(self):
        """Testing fromiter (empty iter)"""
        a = np.array([], dtype="f8")
        b = ca.fromiter(iter(a), dtype='f8')
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test03(self):
        """Testing fromiter (dtype conversion)"""
        a = np.arange(101, dtype="f8")
        b = ca.fromiter(iter(a), dtype='f4')
        assert_array_equal(b[:], a, "fromiter does not work correctly")




def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(chunkTest))
    theSuite.addTest(unittest.makeSuite(carrayTest))
    theSuite.addTest(unittest.makeSuite(copyTest))
    theSuite.addTest(unittest.makeSuite(IterTest))
    theSuite.addTest(unittest.makeSuite(whereTest))
    theSuite.addTest(unittest.makeSuite(getifTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexingTest))
    theSuite.addTest(unittest.makeSuite(fromiterTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
