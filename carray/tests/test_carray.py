# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: August 5, 2010
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

is_64bit = (struct.calcsize("P") == 8)


class chunkTest(unittest.TestCase):

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")


class getitemTest(unittest.TestCase):

    def test01a(self):
        """Testing `__getitem()__` method with only a start"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        sl = slice(1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01b(self):
        """Testing `__getitem()__` method with only a (negative) start"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        sl = slice(-1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01c(self):
        """Testing `__getitem()__` method with only a (start,)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        #print "b[(1,)]->", `b[(1,)]`
        self.assert_(a[(1,)] == b[(1,)], "Values with key (1,) are not equal")

    def test01d(self):
        """Testing `__getitem()__` method with only a (large) start"""
        a = np.arange(1e4)
        b = ca.carray(a)
        sl = -2   # second last element
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02a(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        sl = slice(1, 3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02b(self):
        """Testing `__getitem()__` method with ranges (negative start)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        sl = slice(-3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02c(self):
        """Testing `__getitem()__` method with ranges (negative stop)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(1, -3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02d(self):
        """Testing `__getitem()__` method with ranges (negative start, stop)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(-3, -1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02e(self):
        """Testing `__getitem()__` method with start > stop"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(4, 3, 30)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03a(self):
        """Testing `__getitem()__` method with ranges and steps (I)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(1, 80, 3)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with ranges and steps (II)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(1, 80, 30)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03c(self):
        """Testing `__getitem()__` method with ranges and steps (III)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(990, 998, 2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03d(self):
        """Testing `__getitem()__` method with ranges and steps (IV)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(4, 80, 3000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=100)
        sl = slice(1, 8000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with no start"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=100)
        sl = slice(None, 8000)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with no stop"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=100)
        sl = slice(8000, None)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04d(self):
        """Testing `__getitem()__` method with no start and no stop"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=100)
        sl = slice(None, None, 2)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test05(self):
        """Testing `__getitem()__` method with negative steps"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=10)
        sl = slice(None, None, -3)
        #print "b[sl]->", `b[sl]`
        self.assertRaises(NotImplementedError, b.__getitem__, sl)


class setitemTest(unittest.TestCase):

    def test00a(self):
        """Testing `__setitem()__` method with only one element"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        b[1] = 10.
        a[1] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test00b(self):
        """Testing `__setitem()__` method with only one element (tuple)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        b[(1,)] = 10.
        a[(1,)] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test01(self):
        """Testing `__setitem()__` method with a range"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        b[10:100] = np.arange(1e2 - 10.)
        a[10:100] = np.arange(1e2 - 10.)
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test02(self):
        """Testing `__setitem()__` method with broadcasting"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        b[10:100] = 10.
        a[10:100] = 10.
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test03(self):
        """Testing `__setitem()__` method with the complete range"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=10)
        b[:] = np.arange(10., 1e2 + 10.)
        a[:] = np.arange(10., 1e2 + 10.)
        #print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04a(self):
        """Testing `__setitem()__` method with start:stop:step"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=1)
        sl = slice(10, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04b(self):
        """Testing `__setitem()__` method with start:stop:step (II)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=1)
        sl = slice(10, 11, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04c(self):
        """Testing `__setitem()__` method with start:stop:step (III)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=1)
        sl = slice(96, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04d(self):
        """Testing `__setitem()__` method with start:stop:step (IV)"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=1)
        sl = slice(2, 99, 30)
        b[sl] = 10.
        a[sl] = 10.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test05(self):
        """Testing `__setitem()__` method with negative step"""
        a = np.arange(1e2)
        b = ca.carray(a, chunklen=1)
        sl = slice(2, 99, -30)
        self.assertRaises(NotImplementedError, b.__setitem__, sl, 3.)


class appendTest(unittest.TestCase):

    def test00(self):
        """Testing `append()` method"""
        a = np.arange(1e3)
        b = ca.carray(a)
        b.append(a)
        #print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test01(self):
        """Testing `append()` method (small chunklen)"""
        a = np.arange(1e3)
        b = ca.carray(a, chunklen=1)
        b.append(a)
        #print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `append()` method (large append)"""
        a = np.arange(1e4)
        c = np.arange(2e5)
        b = ca.carray(a)
        b.append(c)
        #print "b->", `b`
        d = np.concatenate((a, c))
        assert_array_equal(d, b[:], "Arrays are not equal")


class trimTest(unittest.TestCase):

    def test00(self):
        """Testing `trim()` method"""
        b = ca.arange(1e3)
        b.trim(3)
        a = np.arange(1e3-3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing `trim()` method (small chunklen)"""
        b = ca.arange(1e2, chunklen=2)
        b.trim(5)
        a = np.arange(1e2-5)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `trim()` method (large trim)"""
        a = np.arange(2)
        b = ca.arange(1e4)
        b.trim(1e4-2)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test03(self):
        """Testing `trim()` method (complete trim)"""
        a = np.arange(0.)
        b = ca.arange(1e4)
        b.trim(1e4)
        #print "b->", `b`
        self.assert_(len(a) == len(b), "Lengths are not equal")

    def test04(self):
        """Testing `trim()` method (trimming more than available items)"""
        a = np.arange(0.)
        b = ca.arange(1e4)
        #print "b->", `b`
        self.assertRaises(ValueError, b.trim, 1e4+1)

    def test05(self):
        """Testing `trim()` method (trimming zero items)"""
        a = np.arange(1e1)
        b = ca.arange(1e1)
        b.trim(0)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test06(self):
        """Testing `trim()` method (negative number of items)"""
        a = np.arange(2e1)
        b = ca.arange(1e1)
        b.trim(-10)
        a[10:] = 0
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")


class resizeTest(unittest.TestCase):

    def test00a(self):
        """Testing `resize()` method (decrease)"""
        b = ca.arange(self.N)
        b.resize(self.N-3)
        a = np.arange(self.N-3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test00b(self):
        """Testing `resize()` method (increase)"""
        b = ca.arange(self.N)
        b.resize(self.N+3)
        a = np.arange(self.N+3)
        a[self.N:] = 0
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01a(self):
        """Testing `resize()` method (decrease, large variation)"""
        b = ca.arange(self.N)
        b.resize(3)
        a = np.arange(3)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01b(self):
        """Testing `resize()` method (increase, large variation)"""
        b = ca.arange(self.N, dflt=1)
        b.resize(self.N*3)
        a = np.arange(self.N*3)
        a[self.N:] = 1
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `resize()` method (zero size)"""
        b = ca.arange(self.N)
        b.resize(0)
        a = np.arange(0)
        #print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")


class resize_smallTest(resizeTest):
    N = 10

class resize_largeTest(resizeTest):
    N = 10000

class miscTest(unittest.TestCase):

    def test00(self):
        """Testing __len__()"""
        a = np.arange(111)
        b = ca.carray(a)
        self.assert_(len(a) == len(b), "Arrays do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big carrays)"""
        a = np.arange(2e5)
        b = ca.carray(a)
        #print "size b uncompressed-->", b.nbytes
        #print "size b compressed  -->", b.cbytes
        self.assert_(sys.getsizeof(b) < b.nbytes,
                     "carray does not seem to compress at all")

    def test02(self):
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


class iterTest(unittest.TestCase):

    def test00(self):
        """Testing `iter()` method"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter1->", sum(b)
        #print "sum iter2->", sum((v for v in b))
        self.assert_(sum(a) == sum(b), "Sums are not equal")
        self.assert_(sum((v for v in a)) == sum((v for v in b)),
                     "Sums are not equal")

    def test01a(self):
        """Testing `iter()` method with a positive start"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(3))
        self.assert_(sum(a[3:]) == sum(b.iter(3)), "Sums are not equal")

    def test01b(self):
        """Testing `iter()` method with a negative start"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(-3))
        self.assert_(sum(a[-3:]) == sum(b.iter(-3)), "Sums are not equal")

    def test02a(self):
        """Testing `iter()` method with positive start, stop"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(3, 24))
        self.assert_(sum(a[3:24]) == sum(b.iter(3, 24)), "Sums are not equal")

    def test02b(self):
        """Testing `iter()` method with negative start, stop"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(-24, -3))
        self.assert_(sum(a[-24:-3]) == sum(b.iter(-24, -3)),
                     "Sums are not equal")

    def test02c(self):
        """Testing `iter()` method with positive start, negative stop"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(24, -3))
        self.assert_(sum(a[24:-3]) == sum(b.iter(24, -3)),
                     "Sums are not equal")

    def test03a(self):
        """Testing `iter()` method with only step"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(step=4))
        self.assert_(sum(a[::4]) == sum(b.iter(step=4)),
                     "Sums are not equal")

    def test03b(self):
        """Testing `iter()` method with start, stop, step"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        #print "sum iter->", sum(b.iter(3, 24, 4))
        self.assert_(sum(a[3:24:4]) == sum(b.iter(3, 24, 4)),
                     "Sums are not equal")

    def test03c(self):
        """Testing `iter()` method with negative step"""
        a = np.arange(101)
        b = ca.carray(a, chunklen=2)
        self.assertRaises(NotImplementedError, b.iter, 0, 1, -3)

    def test04(self):
        """Testing `iter()` method with large zero arrays"""
        a = np.zeros(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b), dtype='f8', count=len(a))
        #print "c ->", repr(c)
        assert_array_equal(a, c[:], "iterator fails on zeros")

    def test05(self):
        """Testing `iter()` method with `limit`"""
        a = np.arange(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b.iter(limit=1010)), dtype='f8',
                        count=1010)
        #print "c ->", repr(c)
        assert_array_equal(a[:1010], c, "iterator fails on zeros")

    def test06a(self):
        """Testing `iter()` method with `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b.iter(skip=1010)), dtype='f8',
                        count=10000-1010)
        #print "c ->", repr(c)
        assert_array_equal(a[1010:], c, "iterator fails on zeros")

    def test06b(self):
        """Testing `iter()` method with negative `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b.iter(skip=-1010)), dtype='f8',
                        count=1010)
        #print "c ->", repr(c)
        assert_array_equal(a[-1010:], c, "iterator fails on zeros")

    def test07a(self):
        """Testing `iter()` method with `limit` and `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b.iter(limit=1010, skip=1010)), dtype='f8',
                        count=1010)
        #print "c ->", repr(c)
        assert_array_equal(a[1010:2020], c, "iterator fails on zeros")

    def test07b(self):
        """Testing `iter()` method with `limit` and negative `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = ca.carray(a, chunklen=100)
        c = ca.fromiter((v for v in b.iter(limit=1000, skip=-1010)), dtype='f8',
                        count=1000)
        #print "c ->", repr(c)
        assert_array_equal(a[-1010:-10], c, "iterator fails on zeros")


class wheretrueTest(unittest.TestCase):

    def test00(self):
        """Testing `wheretrue()` iterator (all true values)"""
        a = np.arange(1, 11) > 0
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test01(self):
        """Testing `wheretrue()` iterator (all false values)"""
        a = np.arange(1, 11) < 0
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test02(self):
        """Testing `wheretrue()` iterator (all false values, large array)"""
        a = np.arange(1, 1e5) < 0
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test03(self):
        """Testing `wheretrue()` iterator (mix of true/false values)"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        #print "numpy ->", a.nonzero()[0].tolist()
        #print "where ->", [i for i in b.wheretrue()]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test04(self):
        """Testing `wheretrue()` iterator with `limit`"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()[:3]
        cwt = [i for i in b.wheretrue(limit=3)]
        #print "numpy ->", a.nonzero()[0].tolist()[:3]
        #print "where ->", [i for i in b.wheretrue(limit=3)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test05a(self):
        """Testing `wheretrue()` iterator with `skip`"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()[2:]
        cwt = [i for i in b.wheretrue(skip=2)]
        #print "numpy ->", a.nonzero()[0].tolist()[2:]
        #print "where ->", [i for i in b.wheretrue(skip=2)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test05b(self):
        """Testing `wheretrue()` iterator with negative `skip`"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()[-2:]
        cwt = [i for i in b.wheretrue(skip=-2)]
        #print "numpy ->", a.nonzero()[0].tolist()[-2:]
        #print "where ->", [i for i in b.wheretrue(skip=-2)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test06a(self):
        """Testing `wheretrue()` iterator with `limit` and `skip`"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()[2:4]
        cwt = [i for i in b.wheretrue(limit=2, skip=2)]
        #print "numpy ->", a.nonzero()[0].tolist()[2:4]
        #print "where ->", [i for i in b.wheretrue(limit=2, skip=2)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test06b(self):
        """Testing `wheretrue()` iterator with `limit` and negative `skip`"""
        a = np.arange(1, 11) > 5
        b = ca.carray(a)
        wt = a.nonzero()[0].tolist()[-4:-2]
        cwt = [i for i in b.wheretrue(limit=2, skip=-4)]
        #print "numpy ->", a.nonzero()[0].tolist()[-4:-2]
        #print "where ->", [i for i in b.wheretrue(limit=2, skip=-4)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")

    def test07(self):
        """Testing `wheretrue()` iterator with `limit` and `skip` (zeros)"""
        a = np.arange(10000) > 5000
        b = ca.carray(a, chunklen=100)
        wt = a.nonzero()[0].tolist()[1020:2040]
        cwt = [i for i in b.wheretrue(skip=1020, limit=1020)]
        # print "numpy ->", a.nonzero()[0].tolist()[1020:2040]
        # print "where ->", [i for i in b.wheretrue(limit=1020,skip=1020)]
        self.assert_(wt == cwt, "wheretrue() does not work correctly")


class whereTest(unittest.TestCase):

    def test00(self):
        """Testing `where()` iterator (all true values)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v>0]
        cwt = [v for v in b.where(a>0)]
        #print "numpy ->", [v for v in a if v>0]
        #print "where ->", [v for v in b.where(a>0)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test01(self):
        """Testing `where()` iterator (all false values)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<0]
        cwt = [v for v in b.where(a<0)]
        #print "numpy ->", [v for v in a if v<0]
        #print "where ->", [v for v in b.where(a<0)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02a(self):
        """Testing `where()` iterator (mix of true/false values, I)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.where(a<=5)]
        #print "numpy ->", [v for v in a if v<=5]
        #print "where ->", [v for v in b.where(a<=5)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02b(self):
        """Testing `where()` iterator (mix of true/false values, II)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5 and v>2]
        cwt = [v for v in b.where((a<=5) & (a>2))]
        #print "numpy ->", [v for v in a if v<=5 and v>2]
        #print "where ->", [v for v in b.where((a<=5) & (a>2))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test02c(self):
        """Testing `where()` iterator (mix of true/false values, III)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5 or v>8]
        cwt = [v for v in b.where((a<=5) | (a>8))]
        #print "numpy ->", [v for v in a if v<=5 or v>8]
        #print "where ->", [v for v in b.where((a<=5) | (a>8))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test03(self):
        """Testing `where()` iterator (using a boolean carray)"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5]
        cwt = [v for v in b.where(ca.carray(a<=5))]
        #print "numpy ->", [v for v in a if v<=5]
        #print "where ->", [v for v in b.where(ca.carray(a<=5))]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test04(self):
        """Testing `where()` iterator using `limit`"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5][:3]
        cwt = [v for v in b.where(ca.carray(a<=5), limit=3)]
        #print "numpy ->", [v for v in a if v<=5][:3]
        #print "where ->", [v for v in b.where(ca.carray(a<=5), limit=3)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test05(self):
        """Testing `where()` iterator using `skip`"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5][2:]
        cwt = [v for v in b.where(ca.carray(a<=5), skip=2)]
        #print "numpy ->", [v for v in a if v<=5][2:]
        #print "where ->", [v for v in b.where(ca.carray(a<=5), skip=2)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator using `limit` and `skip`"""
        a = np.arange(1, 11)
        b = ca.carray(a)
        wt = [v for v in a if v<=5][1:4]
        cwt = [v for v in b.where(ca.carray(a<=5), limit=3, skip=1)]
        #print "numpy ->", [v for v in a if v<=5][1:4]
        #print "where ->", [v for v in b.where(ca.carray(a<=5),
        #                                      limit=3, skip=1)]
        self.assert_(wt == cwt, "where() does not work correctly")

    def test07(self):
        """Testing `where()` iterator using `limit` and `skip` (zeros)"""
        a = np.arange(10000)
        b = ca.carray(a,)
        wt = [v for v in a if v<=5000][1010:2020]
        cwt = [v for v in b.where(ca.carray(a<=5000, chunklen=100),
                                  limit=1010, skip=1010)]
        # print "numpy ->", [v for v in a if v>=5000][1010:2020]
        # print "where ->", [v for v in b.where(ca.carray(a>=5000,chunklen=100),
        #                                       limit=1010, skip=1010)]
        self.assert_(wt == cwt, "where() does not work correctly")


class fancy_indexing_getitemTest(unittest.TestCase):

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
        self.assertRaises(IndexError, b.__getitem__, idx)

    def test05(self):
        """Testing `where()` iterator (using bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = ca.carray(a, chunklen=10)
        wt = a[a<5]
        cwt = b[a<5]
        #print "numpy ->", a[a<5]
        #print "where ->", b[a<5]
        assert_array_equal(wt, cwt, "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator (using carray bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = ca.carray(a, chunklen=10)
        wt = a[(a<5)|(a>9)]
        cwt = b[ca.carray((a<5)|(a>9))]
        #print "numpy ->", a[(a<5)|(a>9)]
        #print "where ->", b[ca.carray((a<5)|(a>9))]
        assert_array_equal(wt, cwt, "where() does not work correctly")


class fancy_indexing_setitemTest(unittest.TestCase):

    def test00(self):
        """Testing fancy indexing with __setitem__ (small values)"""
        a = np.arange(1,111)
        b = ca.carray(a, chunklen=10)
        sl = [3, 1]
        b[sl] = (10, 20)
        a[sl] = (10, 20)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing with __setitem__ (large values)"""
        a = np.arange(1,1e3)
        b = ca.carray(a, chunklen=10)
        sl = [0, 300, 998]
        b[sl] = (5, 10, 20)
        a[sl] = (5, 10, 20)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing with __setitem__ (large list)"""
        a = np.arange(0,1000)
        b = ca.carray(a, chunklen=10)
        sl = np.random.randint(0, 1000, size=3*30)
        vals = np.random.randint(1, 1000, size=3*30)
        b[sl] = vals
        a[sl] = vals
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing with __setitem__ (bool array)"""
        a = np.arange(1,1e2)
        b = ca.carray(a, chunklen=10)
        sl = a > 5
        b[sl] = 3.
        a[sl] = 3.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test04(self):
        """Testing fancy indexing with __setitem__ (bool carray)"""
        a = np.arange(1,1e2)
        b = ca.carray(a, chunklen=10)
        bc = (a > 5) & (a < 40)
        sl = ca.carray(bc)
        b[sl] = 3.
        a[bc] = 3.
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test05(self):
        """Testing fancy indexing with __setitem__ (bool, value not scalar)"""
        a = np.arange(1,1e2)
        b = ca.carray(a, chunklen=10)
        sl = a < 5
        b[sl] = range(6, 10)
        a[sl] = range(6, 10)
        #print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")


class fromiterTest(unittest.TestCase):

    def test00(self):
        """Testing fromiter (short iter)"""
        a = np.arange(1,111)
        b = ca.fromiter(iter(a), dtype='i4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test01a(self):
        """Testing fromiter (long iter)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = ca.fromiter(a, dtype='f8', count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test01b(self):
        """Testing fromiter (long iter, chunk is multiple of iter length)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = ca.fromiter(a, dtype='f8', chunklen=1000, count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test02(self):
        """Testing fromiter (empty iter)"""
        a = np.array([], dtype="f8")
        b = ca.fromiter(iter(a), dtype='f8', count=-1)
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test03(self):
        """Testing fromiter (dtype conversion)"""
        a = np.arange(101, dtype="f8")
        b = ca.fromiter(iter(a), dtype='f4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test04a(self):
        """Testing fromiter method with large iterator"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8')
        b = ca.fromiter((i*2 for i in xrange(N)), dtype='f8', count=len(a))
        assert_array_equal(b[:], a, "iterator with a hint fails")

    def test04b(self):
        """Testing fromiter method with large iterator with a hint"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        b = ca.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        assert_array_equal(b[:], a, "iterator with a hint fails")


class evalTest(unittest.TestCase):

    vm = "python"

    def setUp(self):
        self.prev_vm = ca.defaults.eval_vm
        ca.defaults.eval_vm = self.vm

    def tearDown(self):
        ca.defaults.eval_vm = self.prev_vm

    def test00(self):
        """Testing eval() with only scalars and constants"""
        a = 3
        cr = ca.eval("2 * a")
        #print "ca.eval ->", cr
        self.assert_(cr == 6, "eval does not work correctly")

    def test01a(self):
        """Testing eval() with only carrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        cr = ca.eval("c * d")
        nr = a * b
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01b(self):
        """Testing eval() with `user_dict`"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        cr = ca.eval("c * d2", user_dict={'d2': d})
        nr = a * b
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01c(self):
        """Testing eval() with `depth`"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        def compute(expression):
            return ca.eval("c * d", depth=3)
        cr = compute("c * d")
        nr = a * b
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing eval() with only ndarrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        cr = ca.eval("a * b")
        nr = a * b
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing eval() with a mix of carrays and ndarrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        cr = ca.eval("a * d")
        nr = a * b
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing eval() with a mix of carray, ndarray and scalars"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        cr = ca.eval("a + 2 * d - 3")
        nr = a + 2 * b - 3
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing eval() with a mix of carray, ndarray and scalars"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), b
        cr = ca.eval("a + 2 * d - 3")
        nr = a + 2 * b - 3
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test06(self):
        """Testing eval() with only scalars and arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), b
        cr = ca.eval("d - 3")
        nr = b - 3
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test07(self):
        """Testing eval() via expression on __getitem__"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), b
        cr = c["a + 2 * d - 3 > 0"]
        nr = a[(a + 2 * b - 3) > 0]
        #print "ca[expr] ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "carray[expr] does not work correctly")

    def test08(self):
        """Testing eval() via expression with lists (raise ValueError)"""
        a, b = range(int(self.N)), range(int(self.N))
        self.assertRaises(ValueError, ca.eval, "a*3", depth=3)
        self.assertRaises(ValueError, ca.eval, "b*3", depth=3)

    def test09(self):
        """Testing eval() via expression on __setitem__ (I)"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), b
        c["a + 2 * d - 3 > 0"] = 3
        a[(a + 2 * b - 3) > 0] = 3
        #print "carray ->", c
        #print "numpy  ->", a
        assert_array_equal(c[:], a, "carray[expr] = v does not work correctly")

    def test10(self):
        """Testing eval() via expression on __setitem__ (II)"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), b
        c["a + 2 * d - 3 > 1000"] = 0
        a[(a + 2 * b - 3) > 1000] = 0
        #print "carray ->", c
        #print "numpy  ->", a
        assert_array_equal(c[:], a, "carray[expr] = v does not work correctly")

    def test11(self):
        """Testing eval() with functions like `np.sin()`"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        if self.vm == "python":
            cr = ca.eval("np.sin(c) + 2 * np.log(d) - 3")
        else:
            cr = ca.eval("sin(c) + 2 * log(d) - 3")
        nr = np.sin(a) + 2 * np.log(b) - 3
        #print "ca.eval ->", cr
        #print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test12(self):
        """Testing eval() with `out_flavor` == 'numpy'"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = ca.carray(a), ca.carray(b)
        cr = ca.eval("c + 2 * d - 3", out_flavor='numpy')
        nr = a + 2 * b - 3
        #print "ca.eval ->", cr, type(cr)
        #print "numpy   ->", nr
        self.assert_(type(cr) == np.ndarray)
        assert_array_equal(cr, nr, "eval does not work correctly")

class eval_small(evalTest):
    N = 10

class eval_big(evalTest):
    N = 1e4

class eval_small_ne(evalTest):
    N = 10
    vm = "numexpr"

class eval_big_ne(evalTest):
    N = 1e4
    vm = "numexpr"


class computeMethodsTest(unittest.TestCase):

    def test00(self):
        """Testing sum()."""
        a = np.arange(1e5)
        sa = a.sum()
        ac = ca.carray(a)
        sac = ac.sum()
        #print "numpy sum-->", sa
        #print "carray sum-->", sac
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")

    def test01(self):
        """Testing sum() with dtype."""
        a = np.arange(1e5)
        sa = a.sum(dtype='i8')
        ac = ca.carray(a)
        sac = ac.sum(dtype='i8')
        #print "numpy sum-->", sa
        #print "carray sum-->", sac
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")

    def test02(self):
        """Testing sum() with strings (TypeError)."""
        ac = ca.zeros(10, 'S3')
        self.assertRaises(TypeError, ac.sum)


class arangeTest(unittest.TestCase):

    def test00(self):
        """Testing arange() with only a `stop`."""
        a = np.arange(self.N)
        ac = ca.arange(self.N)
        self.assert_(np.all(a == ac))

    def test01(self):
        """Testing arange() with a `start` and `stop`."""
        a = np.arange(3, self.N)
        ac = ca.arange(3, self.N)
        self.assert_(np.all(a == ac))

    def test02(self):
        """Testing arange() with a `start`, `stop` and `step`."""
        a = np.arange(3, self.N, 4)
        ac = ca.arange(3, self.N, 4)
        self.assert_(np.all(a == ac))

    def test03(self):
        """Testing arange() with a `dtype`."""
        a = np.arange(self.N, dtype="i1")
        ac = ca.arange(self.N, dtype="i1")
        self.assert_(np.all(a == ac))

class arange_smallTest(arangeTest):
    N = 10

class arange_bigTest(arangeTest):
    N = 1e4


class constructorTest(unittest.TestCase):

    def test00(self):
        """Testing carray constructor with an int32 `dtype`."""
        a = np.arange(self.N)
        ac = ca.carray(a, dtype='i4')
        self.assert_(ac.dtype == np.dtype('i4'))
        a = a.astype('i4')
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test01a(self):
        """Testing zeros() constructor."""
        a = np.zeros(self.N)
        ac = ca.zeros(self.N)
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test01b(self):
        """Testing zeros() constructor, with a `dtype`."""
        a = np.zeros(self.N, dtype='i4')
        ac = ca.zeros(self.N, dtype='i4')
        #print "dtypes-->", a.dtype, ac.dtype
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test01c(self):
        """Testing zeros() constructor, with a string type."""
        a = np.zeros(self.N, dtype='S5')
        ac = ca.zeros(self.N, dtype='S5')
        #print "ac-->", `ac`
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test02a(self):
        """Testing fill() constructor."""
        a = np.ones(self.N)
        ac = ca.ones(self.N)
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test02b(self):
        """Testing fill() constructor, with a `dtype`."""
        a = np.ones(self.N, dtype='i4')
        ac = ca.ones(self.N, dtype='i4')
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test02c(self):
        """Testing fill() constructor, with a string type"""
        a = np.ones(self.N, dtype='S3')
        ac = ca.ones(self.N, dtype='S3')
        #print "a-->", a, ac
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test03a(self):
        """Testing fill() constructor."""
        a = np.ones(self.N)
        ac = ca.fill(self.N, 1)
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test03b(self):
        """Testing fill() constructor, with a `dtype`."""
        a = np.ones(self.N, dtype='i4')*3
        ac = ca.fill(self.N, 3, dtype='i4')
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test03c(self):
        """Testing fill() constructor, with a string type"""
        a = np.ones(self.N, dtype='S3')
        ac = ca.fill(self.N, "1", dtype='S3')
        #print "a-->", a, ac
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))


class constructor_smallTest(constructorTest):
    N = 10

class constructor_bigTest(constructorTest):
    N = 10000


class dtypesTest(unittest.TestCase):

    def test00(self):
        """Testing carray constructor with a float32 `dtype`."""
        a = np.arange(10)
        ac = ca.carray(a, dtype='f4')
        self.assert_(ac.dtype == np.dtype('f4'))
        a = a.astype('f4')
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test01(self):
        """Testing carray constructor with a `dtype` with an empty input."""
        a = np.array([], dtype='i4')
        ac = ca.carray([], dtype='f4')
        self.assert_(ac.dtype == np.dtype('f4'))
        a = a.astype('f4')
        self.assert_(a.dtype == ac.dtype)
        self.assert_(np.all(a == ac))

    def test02(self):
        """Testing carray constructor with a plain compound `dtype`."""
        dtype = np.dtype("f4,f8")
        a = np.ones(3, dtype=dtype)
        ac = ca.carray(a, dtype=dtype)
        self.assert_(ac.dtype == dtype)
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac[:], "Arrays are not equal")

    def test03(self):
        """Testing carray constructor with a nested compound `dtype`."""
        dtype = np.dtype([('f1', [('f1', 'i2'), ('f2', 'i4')])])
        a = np.ones(3, dtype=dtype)
        ac = ca.carray(a, dtype=dtype)
        self.assert_(ac.dtype == dtype)
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac[:], "Arrays are not equal")

    def test04(self):
        """Testing carray constructor with a string `dtype`."""
        a = np.array(["ale", "e", "aco"], dtype="S4")
        ac = ca.carray(a, dtype='S4')
        self.assert_(ac.dtype == np.dtype('S4'))
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test05(self):
        """Testing carray constructor with a unicode `dtype`."""
        a = np.array([u"aŀle", u"eñe", u"açò"], dtype="U4")
        ac = ca.carray(a, dtype='U4')
        self.assert_(ac.dtype == np.dtype('U4'))
        self.assert_(a.dtype == ac.dtype)
        #print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test06(self):
        """Testing carray constructor with an object `dtype`."""
        dtype = np.dtype("object")
        a = np.array(["ale", "e", "aco"], dtype=dtype)
        self.assertRaises(TypeError, ca.carray, a)


class streamTest(unittest.TestCase):
    
    def test00(self):
        """Testing carray streaming with medium array"""
        c = ca.arange(1e5, dtype=int)
        s = c.tostring()
        assert_array_equal(ca.fromstring(s), c)

    def test01(self):
        """Testing carray streaming with multiple dimensions"""
        # dimensions >2 and indexing is not fully supported by carray right now..
        #shape = (27,27,729)
        shape = (27,56)
        c = ca.zeros(shape, dtype='f4')
        for i in range(10):
            c[i,i] = float(i)
        s = c.tostring()
        assert_array_equal(ca.fromstring(s), c)
    
    def test03(self):
        """Testing tostring/fromstring for carray"""
        c = ca.carray(["ale", "e", "aco"], dtype='S4')
        s = c.tostring()
        assert_array_equal(ca.fromstring(s), c)

class largeCarrayTest(unittest.TestCase):

    def test00(self):
        """Creating and working with an extremely large carray (> 2**32)."""

        cn = ca.zeros(5e9, dtype="i1")
        self.assert_(len(cn) == int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assert_(cn[1] == 1)
        cn[int(2e9)] = 2
        self.assert_(cn[int(2e9)] == 2)
        cn[long(3e9)] = 3
        self.assert_(cn[long(3e9)] == 3)
        cn[-1] = 4
        self.assert_(cn[-1] == 4)

        self.assert_(cn.sum() == 10)


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(chunkTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))
    theSuite.addTest(unittest.makeSuite(setitemTest))
    theSuite.addTest(unittest.makeSuite(appendTest))
    theSuite.addTest(unittest.makeSuite(trimTest))
    theSuite.addTest(unittest.makeSuite(resize_smallTest))
    theSuite.addTest(unittest.makeSuite(resize_largeTest))
    theSuite.addTest(unittest.makeSuite(miscTest))
    theSuite.addTest(unittest.makeSuite(copyTest))
    theSuite.addTest(unittest.makeSuite(iterTest))
    theSuite.addTest(unittest.makeSuite(wheretrueTest))
    theSuite.addTest(unittest.makeSuite(whereTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_getitemTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_setitemTest))
    theSuite.addTest(unittest.makeSuite(fromiterTest))
    theSuite.addTest(unittest.makeSuite(arange_smallTest))
    theSuite.addTest(unittest.makeSuite(arange_bigTest))
    theSuite.addTest(unittest.makeSuite(constructor_smallTest))
    theSuite.addTest(unittest.makeSuite(constructor_bigTest))
    theSuite.addTest(unittest.makeSuite(dtypesTest))
    theSuite.addTest(unittest.makeSuite(streamTest))
    theSuite.addTest(unittest.makeSuite(computeMethodsTest))
    theSuite.addTest(unittest.makeSuite(eval_small))
    theSuite.addTest(unittest.makeSuite(eval_big))
    
    if ca.numexpr_here:
        theSuite.addTest(unittest.makeSuite(eval_small_ne))
        theSuite.addTest(unittest.makeSuite(eval_big_ne))

    # Only for 64-bit systems
    if is_64bit:
        theSuite.addTest(unittest.makeSuite(largeCarrayTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
