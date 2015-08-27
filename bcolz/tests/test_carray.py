# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

import os
import sys
import struct
import shutil
import textwrap
from bcolz.utils import to_ndarray

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from bcolz.tests import common
from bcolz.tests.common import (
    MayBeDiskTest, TestCase, unittest, skipUnless, SkipTest)
import bcolz
from bcolz.py2help import xrange, PY2, _inttypes
from bcolz.carray_ext import chunk
from bcolz import carray
import pickle

import ctypes

is_64bit = (struct.calcsize("P") == 8)

if sys.version_info >= (3, 0):
    long = int


class initTest(TestCase):

    def test_roundtrip_from_transpose1(self):
        """Testing `__init__` called without `dtype` and a non-contiguous (transposed) array."""
        transposed_array = np.array([[0, 1, 2], [2, 1, 0]]).T
        assert_array_equal(transposed_array, carray(transposed_array, dtype=None))

    def test_roundtrip_from_transpose2(self):
        """Testing `__init__` called with `dtype` and a non-contiguous (transposed) array."""
        transposed_array = np.array([[0, 1, 2], [2, 1, 0]]).T
        assert_array_equal(transposed_array, carray(transposed_array, dtype=transposed_array.dtype))

    @unittest.skipIf(not bcolz.pandas_here, "cannot import pandas")
    def test_roundtrip_from_dataframe1(self):
        """Testing `__init__` called without `dtype` and a dataframe over non-contiguous data."""
        import pandas as pd
        df = pd.DataFrame(data={
            'a': np.arange(3),
            'b': np.arange(3)[::-1]
        })
        assert_array_equal(df, carray(df, dtype=None))

    @unittest.skipIf(not bcolz.pandas_here, "cannot import pandas")
    def test_roundtrip_from_dataframe2(self):
        """Testing `__init__` called with `dtype` and a dataframe over non-contiguous data."""
        import pandas as pd
        df = pd.DataFrame(data={
            'a': np.arange(3),
            'b': np.arange(3)[::-1]
        })
        ca = carray(df, dtype=np.dtype(np.float))
        assert_array_equal(df, ca)
        self.assertEqual(ca.dtype, np.dtype(np.float),
                         msg='carray has been created with invalid dtype')

    def test_dtype_None(self):
        """Testing `utils.to_ndarray` called without `dtype` and a non-contiguous (transposed) array."""
        array = np.array([[0, 1, 2], [2, 1, 0]]).T
        self.assertTrue(to_ndarray(array, None, safe=True).flags.contiguous,
                        msg='to_ndarray: Non contiguous arrays are not being consolidated when dtype is None')


class chunkTest(TestCase):

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=bcolz.cparams())
        # print "b[1]->", `b[1]`
        self.assertTrue(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=bcolz.cparams())
        # print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=bcolz.cparams())
        # print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = chunk(a, atom=a.dtype, cparams=bcolz.cparams())
        # print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")


class pickleTest(MayBeDiskTest):

    disk = False

    def generate_data(self):
        return bcolz.arange(1e2)

    def test_pickleable(self):
        b = self.generate_data()
        s = pickle.dumps(b)
        if PY2:
            self.assertIsInstance(s, str)
        else:
            self.assertIsInstance(s, bytes)
        if self.disk:
            b2 = pickle.loads(s)
            # this should probably be self.assertEquals(b, b2)
            # but at the time == didn't work
            self.assertEquals(b2.rootdir, b.rootdir)


class pickleTestDisk(pickleTest):
    disk = True

    def generate_data(self):
        return bcolz.arange(1e2, rootdir=self.rootdir)


class getitemTest(MayBeDiskTest):

    def test01a(self):
        """Testing `__getitem()__` method with only a start"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01b(self):
        """Testing `__getitem()__` method with only a (negative) start"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-1)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test01c(self):
        """Testing `__getitem()__` method with only a (start,)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        # print "b[(1,)]->", `b[(1,)]`
        self.assertTrue(a[(1,)] == b[(1,)],
                        "Values with key (1,) are not equal")

    def test01d(self):
        """Testing `__getitem()__` method with only a (large) start"""
        a = np.arange(1e4)
        b = bcolz.carray(a, rootdir=self.rootdir)
        sl = -2   # second last element
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02a(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 3)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02b(self):
        """Testing `__getitem()__` method with ranges (negative start)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-3)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02c(self):
        """Testing `__getitem()__` method with ranges (negative stop)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, -3)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02d(self):
        """Testing `__getitem()__` method with ranges (negative start, stop)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(-3, -1)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test02e(self):
        """Testing `__getitem()__` method with start > stop"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(4, 3, 30)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03a(self):
        """Testing `__getitem()__` method with ranges and steps (I)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 80, 3)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with ranges and steps (II)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 80, 30)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03c(self):
        """Testing `__getitem()__` method with ranges and steps (III)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(990, 998, 2)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03d(self):
        """Testing `__getitem()__` method with ranges and steps (IV)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(4, 80, 3000)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with lsmall ranges"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=16, rootdir=self.rootdir)
        sl = slice(1, 2)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04ab(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(1, 8000)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with no start"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(None, 8000)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with no stop"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(8000, None)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04d(self):
        """Testing `__getitem()__` method with no start and no stop"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        sl = slice(None, None, 2)
        # print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test05(self):
        """Testing `__getitem()__` method with negative steps"""
        a = np.arange(1e3)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        sl = slice(None, None, -3)
        # print "b[sl]->", `b[sl]`
        self.assertRaises(NotImplementedError, b.__getitem__, sl)


class getitemMemoryTest(getitemTest, TestCase):
    disk = False


class getitemDiskTest(getitemTest, TestCase):
    disk = True


class setitemTest(MayBeDiskTest):

    def test00a(self):
        """Testing `__setitem()__` method with only one element"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        b[1] = 10.
        a[1] = 10.
        # print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test00b(self):
        """Testing `__setitem()__` method with only one element (tuple)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        b[(1,)] = 10.
        a[(1,)] = 10.
        # print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test01(self):
        """Testing `__setitem()__` method with a range"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        b[10:100] = np.arange(1e2 - 10.)
        a[10:100] = np.arange(1e2 - 10.)
        # print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test02(self):
        """Testing `__setitem()__` method with broadcasting"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        b[10:100] = 10.
        a[10:100] = 10.
        # print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test03(self):
        """Testing `__setitem()__` method with the complete range"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=10, rootdir=self.rootdir)
        b[:] = np.arange(10., 1e2 + 10.)
        a[:] = np.arange(10., 1e2 + 10.)
        # print "b->", `b`
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04a(self):
        """Testing `__setitem()__` method with start:stop:step"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(10, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04b(self):
        """Testing `__setitem()__` method with start:stop:step (II)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(10, 11, 3)
        b[sl] = 10.
        a[sl] = 10.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04c(self):
        """Testing `__setitem()__` method with start:stop:step (III)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(96, 100, 3)
        b[sl] = 10.
        a[sl] = 10.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test04d(self):
        """Testing `__setitem()__` method with start:stop:step (IV)"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(2, 99, 30)
        b[sl] = 10.
        a[sl] = 10.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(a, b[:], "__setitem__ not working correctly")

    def test05(self):
        """Testing `__setitem()__` method with negative step"""
        a = np.arange(1e2)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        sl = slice(2, 99, -30)
        self.assertRaises(NotImplementedError, b.__setitem__, sl, 3.)


class setitemMemoryTest(setitemTest, TestCase):
    disk = False


class setitemDiskTest(setitemTest, TestCase):
    disk = True


class appendTest(MayBeDiskTest):

    def test00(self):
        """Testing `append()` method"""
        a = np.arange(1000)
        b = bcolz.carray(a, rootdir=self.rootdir)
        b.append(a)
        # print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test01(self):
        """Testing `append()` method (small chunklen)"""
        a = np.arange(1000)
        b = bcolz.carray(a, chunklen=1, rootdir=self.rootdir)
        b.append(a)
        # print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test02a(self):
        """Testing `append()` method (large chunklen I)"""
        a = np.arange(1000)
        b = bcolz.carray(a, chunklen=10*1000, rootdir=self.rootdir)
        b.append(a)
        # print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test02b(self):
        """Testing `append()` method (large chunklen II)"""
        a = np.arange(100*1000)
        b = bcolz.carray(a, chunklen=10*1000, rootdir=self.rootdir)
        b.append(a)
        # print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test02c(self):
        """Testing `append()` method (large chunklen III)"""
        a = np.arange(1000*1000)
        b = bcolz.carray(a, chunklen=100*1000-1, rootdir=self.rootdir)
        b.append(a)
        # print "b->", `b`
        c = np.concatenate((a, a))
        assert_array_equal(c, b[:], "Arrays are not equal")

    def test03(self):
        """Testing `append()` method (large append)"""
        a = np.arange(1e4)
        c = np.arange(2e5)
        b = bcolz.carray(a, rootdir=self.rootdir)
        b.append(c)
        # print "b->", `b`
        d = np.concatenate((a, c))
        assert_array_equal(d, b[:], "Arrays are not equal")


class appendMemoryTest(appendTest, TestCase):
    disk = False


class appendDiskTest(appendTest, TestCase):
    disk = True


class trimTest(MayBeDiskTest):

    def test00(self):
        """Testing `trim()` method"""
        b = bcolz.arange(1e3, rootdir=self.rootdir)
        b.trim(3)
        a = np.arange(1e3-3)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing `trim()` method (small chunklen)"""
        b = bcolz.arange(1e2, chunklen=2, rootdir=self.rootdir)
        b.trim(5)
        a = np.arange(1e2-5)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `trim()` method (large trim)"""
        a = np.arange(2)
        b = bcolz.arange(1e4, rootdir=self.rootdir)
        b.trim(1e4-2)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test03(self):
        """Testing `trim()` method (complete trim)"""
        a = np.arange(0.)
        b = bcolz.arange(1e4, rootdir=self.rootdir)
        b.trim(1e4)
        # print "b->", `b`
        self.assertTrue(len(a) == len(b), "Lengths are not equal")

    def test04(self):
        """Testing `trim()` method (trimming more than available items)"""
        a = np.arange(0.)
        b = bcolz.arange(1e4, rootdir=self.rootdir)
        # print "b->", `b`
        self.assertRaises(ValueError, b.trim, 1e4+1)

    def test05(self):
        """Testing `trim()` method (trimming zero items)"""
        a = np.arange(1e1)
        b = bcolz.arange(1e1, rootdir=self.rootdir)
        b.trim(0)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test06(self):
        """Testing `trim()` method (negative number of items)"""
        a = np.arange(2e1)
        b = bcolz.arange(1e1, rootdir=self.rootdir)
        b.trim(-10)
        a[10:] = 0
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")


class trimMemoryTest(trimTest, TestCase):
    disk = False


class trimDiskTest(trimTest, TestCase):
    disk = True


class resizeTest(MayBeDiskTest):

    def test00a(self):
        """Testing `resize()` method (decrease)"""
        b = bcolz.arange(self.N, rootdir=self.rootdir)
        b.resize(self.N-3)
        a = np.arange(self.N-3)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test00b(self):
        """Testing `resize()` method (increase)"""
        b = bcolz.arange(self.N, rootdir=self.rootdir)
        b.resize(self.N+3)
        a = np.arange(self.N+3)
        a[self.N:] = 0
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01a(self):
        """Testing `resize()` method (decrease, large variation)"""
        b = bcolz.arange(self.N, rootdir=self.rootdir)
        b.resize(3)
        a = np.arange(3)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01b(self):
        """Testing `resize()` method (increase, large variation)"""
        b = bcolz.arange(self.N, dflt=1, rootdir=self.rootdir)
        b.resize(self.N*3)
        a = np.arange(self.N*3)
        a[self.N:] = 1
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing `resize()` method (zero size)"""
        b = bcolz.arange(self.N, rootdir=self.rootdir)
        b.resize(0)
        a = np.arange(0)
        # print "b->", `b`
        assert_array_equal(a, b[:], "Arrays are not equal")


class resize_smallTest(resizeTest, TestCase):
    N = 10


class resize_smallDiskTest(resizeTest, TestCase):
    N = 10
    disk = True


class resize_largeTest(resizeTest, TestCase):
    N = 10000


class resize_largeDiskTest(resizeTest, TestCase):
    N = 10000
    disk = True


class miscTest(MayBeDiskTest):

    def test00(self):
        """Testing __len__()"""
        a = np.arange(111)
        b = bcolz.carray(a, rootdir=self.rootdir)
        self.assertTrue(len(a) == len(b), "Arrays do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big carrays)"""
        a = np.arange(2e5)
        b = bcolz.carray(a, rootdir=self.rootdir)
        # print "size b uncompressed-->", b.nbytes
        # print "size b compressed  -->", b.cbytes
        self.assertTrue(sys.getsizeof(b) < b.nbytes,
                        "carray does not seem to compress at all")

    def test02(self):
        """Testing __sizeof__() (small carrays)"""
        a = np.arange(111)
        b = bcolz.carray(a)
        # print "size b uncompressed-->", b.nbytes
        # print "size b compressed  -->", b.cbytes
        self.assertTrue(sys.getsizeof(b) > b.nbytes,
                        "carray compress too much??")


class miscMemoryTest(miscTest, TestCase):
    disk = False


class miscDiskTest(miscTest, TestCase):
    disk = True


class copyTest(MayBeDiskTest):

    N = int(1e5)

    def tearDown(self):
        # Restore defaults
        bcolz.cparams.setdefaults(clevel=5, shuffle=True, cname='blosclz')
        MayBeDiskTest.tearDown(self)

    def test00(self):
        """Testing copy() without params"""
        a = np.arange(111)
        b = bcolz.carray(a, rootdir=self.rootdir)
        c = b.copy()
        c.append(np.arange(111, 122))
        self.assertTrue(len(b) == 111, "copy() does not work well")
        self.assertTrue(len(c) == 122, "copy() does not work well")
        r = np.arange(122)
        assert_array_equal(c[:], r, "incorrect correct values after copy()")

    def test01(self):
        """Testing copy() with higher compression"""
        a = np.linspace(-1., 1., self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        c = b.copy(cparams=bcolz.cparams(clevel=9))
        # print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assertTrue(b.cbytes > c.cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lesser compression"""
        a = np.linspace(-1., 1., self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        bcolz.cparams.setdefaults(clevel=1)
        c = b.copy()
        # print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assertTrue(b.cbytes < c.cbytes, "clevel not changed")

    def test03a(self):
        """Testing copy() with no shuffle"""
        a = np.linspace(-1., 1., self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        bcolz.cparams.setdefaults(clevel=1)
        c = b.copy(cparams=bcolz.cparams(shuffle=False))
        # print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assertTrue(b.cbytes < c.cbytes, "shuffle not changed")

    def test03b(self):
        """Testing copy() with no shuffle (setdefaults version)"""
        a = np.linspace(-1., 1., self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        bcolz.cparams.setdefaults(shuffle=False)
        c = b.copy()
        # print "b.cbytes, c.cbytes:", b.cbytes, c.cbytes
        self.assertTrue(b.cbytes < c.cbytes, "shuffle not changed")


class copyMemoryTest(copyTest, TestCase):
    disk = False


class copyDiskTest(copyTest, TestCase):
    disk = True


class viewTest(MayBeDiskTest):

    def tearDown(self):
        MayBeDiskTest.tearDown(self)

    def test00(self):
        """Testing view()"""
        a = np.arange(self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        c = b.view()
        self.assertEqual(len(c), self.N)
        assert_array_equal(c[:], a)

    def test01(self):
        """Testing view() and appends"""
        a = np.arange(self.N)
        b = bcolz.carray(a, rootdir=self.rootdir)
        c = b.view()
        c.append(np.arange(self.N, self.N + 11))
        self.assertEqual(len(b), self.N)
        self.assertEqual(len(c), self.N + 11)
        r = np.arange(self.N + 11)
        assert_array_equal(b[:], a)
        assert_array_equal(c[:], r)

    def test02(self):
        """Testing view() and iterators"""
        a = np.arange(self.N, dtype='uint64')
        b = bcolz.carray(a, rootdir=self.rootdir)
        c = iter(b.view())
        u = c.iter(3)
        w = b.iter(2)
        self.assertEqual(sum(a[3:]), sum(u))
        self.assertEqual(sum(a[2:]), sum(w))


class small_viewMemoryTest(viewTest, TestCase):
    N = 111
    disk = False


class small_viewDiskTest(viewTest, TestCase):
    N = 111
    disk = True


class large_viewMemoryTest(viewTest, TestCase):
    N = int(1e5)
    disk = False


class large_viewDiskTest(viewTest, TestCase):
    N = int(1e5)
    disk = True


class iterTest(MayBeDiskTest):

    def test00a(self):
        """Testing `next()` method"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        for i in range(101):
            self.assertEqual(i, next(b))

    def test00b(self):
        """Testing `iter()` method"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter1->", sum(b)
        # print "sum iter2->", sum((v for v in b))
        self.assertEqual(sum(a), sum(b))
        self.assertEqual(sum((v for v in a)), sum((v for v in b)))

    def test01a(self):
        """Testing `iter()` method with a positive start"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(3))
        self.assertTrue(sum(a[3:]) == sum(b.iter(3)), "Sums are not equal")

    def test01b(self):
        """Testing `iter()` method with a negative start"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(-3))
        self.assertTrue(sum(a[-3:]) == sum(b.iter(-3)), "Sums are not equal")

    def test02a(self):
        """Testing `iter()` method with positive start, stop"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(3, 24))
        self.assertTrue(sum(a[3:24]) == sum(b.iter(3, 24)),
                        "Sums are not equal")

    def test02b(self):
        """Testing `iter()` method with negative start, stop"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(-24, -3))
        self.assertTrue(sum(a[-24:-3]) == sum(b.iter(-24, -3)),
                        "Sums are not equal")

    def test02c(self):
        """Testing `iter()` method with positive start, negative stop"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(24, -3))
        self.assertTrue(sum(a[24:-3]) == sum(b.iter(24, -3)),
                        "Sums are not equal")

    def test03a(self):
        """Testing `iter()` method with only step"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(step=4))
        self.assertTrue(sum(a[::4]) == sum(b.iter(step=4)),
                        "Sums are not equal")

    def test03b(self):
        """Testing `iter()` method with start, stop, step"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        # print "sum iter->", sum(b.iter(3, 24, 4))
        self.assertTrue(sum(a[3:24:4]) == sum(b.iter(3, 24, 4)),
                        "Sums are not equal")

    def test03c(self):
        """Testing `iter()` method with negative step"""
        a = np.arange(101)
        b = bcolz.carray(a, chunklen=2, rootdir=self.rootdir)
        self.assertRaises(NotImplementedError, b.iter, 0, 1, -3)

    def test04(self):
        """Testing `iter()` method with large zero arrays"""
        a = np.zeros(int(1e4), dtype='f8')
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        c = bcolz.fromiter((v for v in b), dtype='f8', count=len(a))
        # print "c ->", repr(c)
        assert_allclose(a, c[:], err_msg="iterator fails on zeros")

    def test05(self):
        """Testing `iter()` method with `limit`"""
        a = np.arange(1e4, dtype='f8')
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        c = bcolz.fromiter((v for v in b.iter(limit=1010)), dtype='f8',
                           count=1010)
        # print "c ->", repr(c)
        assert_allclose(a[:1010], c, err_msg="iterator fails on zeros")

    def test06(self):
        """Testing `iter()` method with `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        c = bcolz.fromiter((v for v in b.iter(skip=1010)), dtype='f8',
                           count=10000-1010)
        # print "c ->", repr(c)
        assert_allclose(a[1010:], c, err_msg="iterator fails on zeros")

    def test07(self):
        """Testing `iter()` method with `limit` and `skip`"""
        a = np.arange(1e4, dtype='f8')
        b = bcolz.carray(a, chunklen=100, rootdir=self.rootdir)
        c = bcolz.fromiter((v for v in b.iter(limit=1010, skip=1010)),
                           dtype='f8',
                           count=1010)
        # print "c ->", repr(c)
        assert_allclose(a[1010:2020], c, err_msg="iterator fails on zeros")

    def test08a(self):
        """Testing several iterators in stage (I)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, rootdir=self.rootdir)
        u = iter(b)
        w = b.iter(2, 20, 2)
        self.assertEqual(a.tolist(), list(b))
        self.assertEqual(sum(a), sum(u))
        self.assertEqual(sum(a[2:20:2]), sum(w))

    def test08b(self):
        """Testing several iterators in stage (II)"""
        a = np.arange(1e3)
        b = bcolz.carray(a, rootdir=self.rootdir)
        u = b.iter(3, 30, 3)
        w = b.iter(2, 20, 2)
        self.assertEqual(a.tolist(), list(b))
        self.assertEqual(sum(a[3:30:3]), sum(u))
        self.assertEqual(sum(a[2:20:2]), sum(w))

    def test09a(self):
        """Testing several iterators in parallel (I)"""
        a = np.arange(10)
        b = bcolz.carray(a, rootdir=self.rootdir)
        b1 = iter(b)
        b2 = iter(b)
        a1 = iter(a)
        a2 = iter(a)
        # print "result:",  [i for i in zip(b1, b2)]
        self.assertEqual([i for i in zip(a1, a2)], [i for i in zip(b1, b2)])

    def test09b(self):
        """Testing several iterators in parallel (II)"""
        a = np.arange(10)
        b = bcolz.carray(a, rootdir=self.rootdir)
        b1 = b.iter(2, 10, 2)
        b2 = b.iter(1, 5, 1)
        a1 = iter(a[2:10:2])
        a2 = iter(a[1:5:1])
        # print "result:",  [i for i in zip(b1, b2)]
        self.assertEqual([i for i in zip(a1, a2)], [i for i in zip(b1, b2)])

    def test10a(self):
        """Testing the reuse of exhausted iterators (I)"""
        a = np.arange(10)
        b = bcolz.carray(a, rootdir=self.rootdir)
        bi = iter(b)
        ai = iter(a)
        self.assertEqual([i for i in ai], [i for i in bi])
        self.assertEqual([i for i in ai], [i for i in bi])

    def test10b(self):
        """Testing the reuse of exhausted iterators (II)"""
        a = np.arange(10)
        b = bcolz.carray(a, rootdir=self.rootdir)
        bi = b.iter(2, 10, 2)
        ai = iter(a[2:10:2])
        # print "result:", [i for i in bi]
        self.assertEqual([i for i in ai], [i for i in bi])
        self.assertEqual([i for i in ai], [i for i in bi])


class iterMemoryTest(iterTest, TestCase):
    disk = False


class iterDiskTest(iterTest, TestCase):
    disk = True


class iterblocksTest(MayBeDiskTest):

    def test00(self):
        """Testing `iterblocks` method with no blen, no start, no stop"""
        N = self.N
        a = bcolz.fromiter(xrange(N), dtype=np.float64, count=N,
                           rootdir=self.rootdir)
        l, s = 0, 0
        for block in bcolz.iterblocks(a):
            l += len(block)
            s += block.sum()
        self.assertEqual(l, N)
        # as per Gauss summation formula
        self.assertEqual(s, (N - 1) * (N / 2))

    def test01(self):
        """Testing `iterblocks` method with no start, no stop"""
        N, blen = self.N, 100
        a = bcolz.fromiter(xrange(N), dtype=np.float64, count=N,
                           rootdir=self.rootdir)
        l, s = 0, 0
        for block in bcolz.iterblocks(a, blen):
            if l == 0:
                self.assertEqual(len(block), blen)
            l += len(block)
            s += block.sum()
        self.assertEqual(l, N)

    def test02(self):
        """Testing `iterblocks` method with no stop"""
        N, blen = self.N, 100
        a = bcolz.fromiter(xrange(N), dtype=np.float64, count=N,
                           rootdir=self.rootdir)
        l, s = 0, 0
        for block in bcolz.iterblocks(a, blen, blen-1):
            l += len(block)
            s += block.sum()
        self.assertEqual(l, (N - (blen - 1)))
        self.assertEqual(s, np.arange(blen-1, N).sum())

    def test03(self):
        """Testing `iterblocks` method with all parameters set"""
        N, blen = self.N, 100
        a = bcolz.fromiter(xrange(N), dtype=np.float64, count=N,
                           rootdir=self.rootdir)
        l, s = 0, 0
        for block in bcolz.iterblocks(a, blen, blen-1, 3*blen+2):
            l += len(block)
            s += block.sum()
        mlen = min(N - (blen - 1), 2*blen + 3)
        self.assertEqual(l, mlen)
        slen = min(N, 3*blen + 2)
        self.assertEqual(s, np.arange(blen-1, slen).sum())


class small_iterblocksMemoryTest(iterblocksTest, TestCase):
    N = 120
    disk = False


class small_iterblocksDiskTest(iterblocksTest, TestCase):
    N = 120
    disk = True


class large_iterblocksMemoryTest(iterblocksTest, TestCase):
    N = 10000
    disk = False


class large_iterblocksDiskTest(iterblocksTest, TestCase):
    N = 10030
    disk = True


class wheretrueTest(TestCase):

    def test00(self):
        """Testing `wheretrue()` iterator (all true values)"""
        a = np.arange(1, 11) > 0
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        # print "numpy ->", a.nonzero()[0].tolist()
        # print "where ->", [i for i in b.wheretrue()]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test01(self):
        """Testing `wheretrue()` iterator (all false values)"""
        a = np.arange(1, 11) < 0
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        # print "numpy ->", a.nonzero()[0].tolist()
        # print "where ->", [i for i in b.wheretrue()]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test02(self):
        """Testing `wheretrue()` iterator (all false values, large array)"""
        a = np.arange(1, 1e5) < 0
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        # print "numpy ->", a.nonzero()[0].tolist()
        # print "where ->", [i for i in b.wheretrue()]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test03(self):
        """Testing `wheretrue()` iterator (mix of true/false values)"""
        a = np.arange(1, 11) > 5
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()
        cwt = [i for i in b.wheretrue()]
        # print "numpy ->", a.nonzero()[0].tolist()
        # print "where ->", [i for i in b.wheretrue()]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test04(self):
        """Testing `wheretrue()` iterator with `limit`"""
        a = np.arange(1, 11) > 5
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()[:3]
        cwt = [i for i in b.wheretrue(limit=3)]
        # print "numpy ->", a.nonzero()[0].tolist()[:3]
        # print "where ->", [i for i in b.wheretrue(limit=3)]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test05(self):
        """Testing `wheretrue()` iterator with `skip`"""
        a = np.arange(1, 11) > 5
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()[2:]
        cwt = [i for i in b.wheretrue(skip=2)]
        # print "numpy ->", a.nonzero()[0].tolist()[2:]
        # print "where ->", [i for i in b.wheretrue(skip=2)]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test06(self):
        """Testing `wheretrue()` iterator with `limit` and `skip`"""
        a = np.arange(1, 11) > 5
        b = bcolz.carray(a)
        wt = a.nonzero()[0].tolist()[2:4]
        cwt = [i for i in b.wheretrue(skip=2, limit=2)]
        # print "numpy ->", a.nonzero()[0].tolist()[2:4]
        # print "where ->", [i for i in b.wheretrue(limit=2,skip=2)]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test07(self):
        """Testing `wheretrue()` iterator with `limit` and `skip` (zeros)"""
        a = np.arange(10000) > 5000
        b = bcolz.carray(a, chunklen=100)
        wt = a.nonzero()[0].tolist()[1020:2040]
        cwt = [i for i in b.wheretrue(skip=1020, limit=1020)]
        # print "numpy ->", a.nonzero()[0].tolist()[1020:2040]
        # print "where ->", [i for i in b.wheretrue(limit=1020,skip=1020)]
        self.assertTrue(wt == cwt, "wheretrue() does not work correctly")

    def test08(self):
        """Testing several iterators in stage"""
        a = np.arange(10000) > 5000
        b = bcolz.carray(a, chunklen=100)
        u = b.wheretrue(skip=1020, limit=1020)
        w = b.wheretrue(skip=1030, limit=1030)
        self.assertEqual(a.nonzero()[0].tolist()[1020:2040], list(u))
        self.assertEqual(a.nonzero()[0].tolist()[1030:2060], list(w))

    def test09(self):
        """Testing several iterators in parallel"""
        a = np.arange(10000) > 5000
        b = bcolz.carray(a, chunklen=100)
        b1 = b.wheretrue(skip=1020, limit=1020)
        b2 = b.wheretrue(skip=1030, limit=1020)
        a1 = a.nonzero()[0].tolist()[1020:2040]
        a2 = a.nonzero()[0].tolist()[1030:2050]
        # print "result:",  [i for i in zip(b1, b2)]
        self.assertEqual([i for i in zip(a1, a2)], [i for i in zip(b1, b2)])

    def test10(self):
        """Testing the reuse of exhausted iterators"""
        a = np.arange(10000) > 5000
        b = bcolz.carray(a, chunklen=100)
        bi = b.wheretrue(skip=1020, limit=1020)
        ai = iter(np.array(a.nonzero()[0].tolist()[1020:2040]))
        self.assertEqual([i for i in ai], [i for i in bi])
        self.assertEqual([i for i in ai], [i for i in bi])


class whereTest(TestCase):

    def test00(self):
        """Testing `where()` iterator (all true values)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v > 0]
        cwt = [v for v in b.where(a > 0)]
        # print "numpy ->", [v for v in a if v>0]
        # print "where ->", [v for v in b.where(a>0)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test01(self):
        """Testing `where()` iterator (all false values)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v < 0]
        cwt = [v for v in b.where(a < 0)]
        # print "numpy ->", [v for v in a if v<0]
        # print "where ->", [v for v in b.where(a<0)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test02a(self):
        """Testing `where()` iterator (mix of true/false values, I)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5]
        cwt = [v for v in b.where(a <= 5)]
        # print "numpy ->", [v for v in a if v<=5]
        # print "where ->", [v for v in b.where(a<=5)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test02b(self):
        """Testing `where()` iterator (mix of true/false values, II)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5 and v > 2]
        cwt = [v for v in b.where((a <= 5) & (a > 2))]
        # print "numpy ->", [v for v in a if v<=5 and v>2]
        # print "where ->", [v for v in b.where((a<=5) & (a>2))]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test02c(self):
        """Testing `where()` iterator (mix of true/false values, III)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5 or v > 8]
        cwt = [v for v in b.where((a <= 5) | (a > 8))]
        # print "numpy ->", [v for v in a if v<=5 or v>8]
        # print "where ->", [v for v in b.where((a<=5) | (a>8))]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test03(self):
        """Testing `where()` iterator (using a boolean carray)"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5]
        cwt = [v for v in b.where(bcolz.carray(a <= 5))]
        # print "numpy ->", [v for v in a if v<=5]
        # print "where ->", [v for v in b.where(bcolz.carray(a<=5))]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test04(self):
        """Testing `where()` iterator using `limit`"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5][:3]
        cwt = [v for v in b.where(bcolz.carray(a <= 5), limit=3)]
        # print "numpy ->", [v for v in a if v<=5][:3]
        # print "where ->", [v for v in b.where(bcolz.carray(a<=5), limit=3)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test05(self):
        """Testing `where()` iterator using `skip`"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5][2:]
        cwt = [v for v in b.where(bcolz.carray(a <= 5), skip=2)]
        # print "numpy ->", [v for v in a if v<=5][2:]
        # print "where ->", [v for v in b.where(bcolz.carray(a<=5), skip=2)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator using `limit` and `skip`"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        wt = [v for v in a if v <= 5][1:4]
        cwt = [v for v in b.where(bcolz.carray(a <= 5), limit=3, skip=1)]
        # print "numpy ->", [v for v in a if v<=5][1:4]
        # print "where ->", [v for v in b.where(bcolz.carray(a<=5),
        #                                      limit=3, skip=1)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test07(self):
        """Testing `where()` iterator using `limit` and `skip` (zeros)"""
        a = np.arange(10000)
        b = bcolz.carray(a,)
        wt = [v for v in a if v <= 5000][1010:2020]
        cwt = [v for v in b.where(bcolz.carray(a <= 5000, chunklen=100),
                                  limit=1010, skip=1010)]
        # print "numpy ->", [v for v in a if v>=5000][1010:2020]
        # print "where ->", [v for v in b.where(bcolz.carray(a>=5000,
        #                                                    chunklen=100),
        #                                       limit=1010, skip=1010)]
        self.assertTrue(wt == cwt, "where() does not work correctly")

    def test08(self):
        """Testing several iterators in stage"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        ul = [v for v in a if v <= 5]
        u = b.where(a <= 5)
        wl = [v for v in a if v <= 6]
        w = b.where(a <= 6)
        self.assertEqual(ul, list(u))
        self.assertEqual(wl, list(w))

    def test09(self):
        """Testing several iterators in parallel"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        b1 = b.where(a <= 5)
        b2 = b.where(a <= 6)
        a1 = [v for v in a if v <= 5]
        a2 = [v for v in a if v <= 6]
        # print "result:",  [i for i in zip(b1, b2)]
        self.assertEqual([i for i in zip(a1, a2)], [i for i in zip(b1, b2)])

    def test10(self):
        """Testing the reuse of exhausted iterators"""
        a = np.arange(1, 11)
        b = bcolz.carray(a)
        bi = b.where(a <= 5)
        ai = (v for v in a if v <= 5)
        self.assertEqual([i for i in ai], [i for i in bi])
        self.assertEqual([i for i in ai], [i for i in bi])


class fancy_indexing_getitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing (short list)"""
        a = np.arange(1, 111)
        b = bcolz.carray(a)
        c = b[[3, 1]]
        r = a[[3, 1]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing (large list, numpy)"""
        a = np.arange(1, 1e4)
        b = bcolz.carray(a)
        idx = np.random.randint(1000, size=1000)
        c = b[idx]
        r = a[idx]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing (empty list)"""
        a = np.arange(101)
        b = bcolz.carray(a)
        c = b[[]]
        r = a[[]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        a = np.arange(1, 101)
        b = bcolz.carray(a)
        c = b[[1.1, 3.3]]
        r = a[[1.1, 3.3]]
        assert_array_equal(c, r, "fancy indexing does not work correctly")

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1, 101)
        b = bcolz.carray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(IndexError, b.__getitem__, idx)

    def test05(self):
        """Testing `where()` iterator (using bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = bcolz.carray(a, chunklen=10)
        wt = a[a < 5]
        cwt = b[a < 5]
        # print "numpy ->", a[a<5]
        # print "where ->", b[a<5]
        assert_array_equal(wt, cwt, "where() does not work correctly")

    def test06(self):
        """Testing `where()` iterator (using carray bool in fancy indexing)"""
        a = np.arange(1, 110)
        b = bcolz.carray(a, chunklen=10)
        wt = a[(a < 5) | (a > 9)]
        cwt = b[bcolz.carray((a < 5) | (a > 9))]
        # print "numpy ->", a[(a<5)|(a>9)]
        # print "where ->", b[bcolz.carray((a<5)|(a>9))]
        assert_array_equal(wt, cwt, "where() does not work correctly")


class fancy_indexing_setitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing with __setitem__ (small values)"""
        a = np.arange(1, 111)
        b = bcolz.carray(a, chunklen=10)
        sl = [3, 1]
        b[sl] = (10, 20)
        a[sl] = (10, 20)
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test01(self):
        """Testing fancy indexing with __setitem__ (large values)"""
        a = np.arange(1, 1e3)
        b = bcolz.carray(a, chunklen=10)
        sl = [0, 300, 998]
        b[sl] = (5, 10, 20)
        a[sl] = (5, 10, 20)
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test02(self):
        """Testing fancy indexing with __setitem__ (large list)"""
        a = np.arange(0, 1000)
        b = bcolz.carray(a, chunklen=10)
        sl = np.random.randint(0, 1000, size=3*30)
        vals = np.random.randint(1, 1000, size=3*30)
        b[sl] = vals
        a[sl] = vals
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test03(self):
        """Testing fancy indexing with __setitem__ (bool array)"""
        a = np.arange(1, 1e2)
        b = bcolz.carray(a, chunklen=10)
        sl = a > 5
        b[sl] = 3.
        a[sl] = 3.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test04(self):
        """Testing fancy indexing with __setitem__ (bool carray)"""
        a = np.arange(1, 1e2)
        b = bcolz.carray(a, chunklen=10)
        bc = (a > 5) & (a < 40)
        sl = bcolz.carray(bc)
        b[sl] = 3.
        a[bc] = 3.
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")

    def test05(self):
        """Testing fancy indexing with __setitem__ (bool, value not scalar)"""
        a = np.arange(1, 1e2)
        b = bcolz.carray(a, chunklen=10)
        sl = a < 5
        b[sl] = range(6, 10)
        a[sl] = range(6, 10)
        # print "b[%s] -> %r" % (sl, b)
        assert_array_equal(b[:], a, "fancy indexing does not work correctly")


class fromiterTest(TestCase):

    def test00(self):
        """Testing fromiter (short iter)"""
        a = np.arange(1, 111)
        b = bcolz.fromiter(iter(a), dtype='i4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test01a(self):
        """Testing fromiter (long iter)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = bcolz.fromiter(a, dtype='f8', count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test01b(self):
        """Testing fromiter (long iter, chunk is multiple of iter length)"""
        N = 1e4
        a = (i for i in xrange(int(N)))
        b = bcolz.fromiter(a, dtype='f8', chunklen=1000, count=int(N))
        c = np.arange(N)
        assert_array_equal(b[:], c, "fromiter does not work correctly")

    def test02(self):
        """Testing fromiter (empty iter)"""
        a = np.array([], dtype="f8")
        b = bcolz.fromiter(iter(a), dtype='f8', count=-1)
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test03(self):
        """Testing fromiter (dtype conversion)"""
        a = np.arange(101, dtype="f8")
        b = bcolz.fromiter(iter(a), dtype='f4', count=len(a))
        assert_array_equal(b[:], a, "fromiter does not work correctly")

    def test04a(self):
        """Testing fromiter method with large iterator"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8')
        b = bcolz.fromiter((i*2 for i in xrange(N)), dtype='f8', count=len(a))
        assert_array_equal(b[:], a, "iterator with a hint fails")

    def test04b(self):
        """Testing fromiter method with large iterator with a hint"""
        N = 10*1000
        a = np.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        b = bcolz.fromiter((i*2 for i in xrange(N)), dtype='f8', count=N)
        assert_array_equal(b[:], a, "iterator with a hint fails")


class evalTest(MayBeDiskTest):

    vm = "python"

    def setUp(self):
        self.prev_vm = bcolz.defaults.eval_vm
        if bcolz.numexpr_here:
            bcolz.defaults.eval_vm = self.vm
        else:
            bcolz.defaults.eval_vm = "python"
        MayBeDiskTest.setUp(self)

    def tearDown(self):
        bcolz.defaults.eval_vm = self.prev_vm
        MayBeDiskTest.tearDown(self)

    def test00(self):
        """Testing eval() with only scalars and constants"""
        a = 3
        cr = bcolz.eval("2 * a", rootdir=self.rootdir)
        # print "bcolz.eval ->", cr
        self.assertTrue(cr == 6, "eval does not work correctly")

    def test01(self):
        """Testing eval() with only carrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        if self.rootdir:
            dirc, dird = self.rootdir+'.c', self.rootdir+'.d'
        else:
            dirc, dird = None, None
        c = bcolz.carray(a, rootdir=dirc)
        d = bcolz.carray(b, rootdir=dird)
        cr = bcolz.eval("c * d")
        nr = a * b
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing eval() with only ndarrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        cr = bcolz.eval("a * b", rootdir=self.rootdir)
        nr = a * b
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing eval() with a mix of carrays and ndarrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        if self.rootdir:
            dirc, dird = self.rootdir+'.c', self.rootdir+'.d'
        else:
            dirc, dird = None, None
        c = bcolz.carray(a, rootdir=dirc)
        d = bcolz.carray(b, rootdir=dird)
        cr = bcolz.eval("a * d")
        nr = a * b
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing eval() with a mix of carray, ndarray and scalars"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        if self.rootdir:
            dirc, dird = self.rootdir+'.c', self.rootdir+'.d'
        else:
            dirc, dird = None, None
        c = bcolz.carray(a, rootdir=dirc)
        d = bcolz.carray(b, rootdir=dird)
        cr = bcolz.eval("a + 2 * d - 3")
        nr = a + 2 * b - 3
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing eval() with a mix of carray, ndarray and scalars"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), b
        cr = bcolz.eval("a + 2 * d - 3")
        nr = a + 2 * b - 3
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test06(self):
        """Testing eval() with only scalars and arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), b
        cr = bcolz.eval("d - 3")
        nr = b - 3
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test07(self):
        """Testing eval() via expression on __getitem__"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), b
        cr = c["a + 2 * d - 3 > 0"]
        nr = a[(a + 2 * b - 3) > 0]
        # print "ca[expr] ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "carray[expr] does not work correctly")

    def test08(self):
        """Testing eval() via expression with lists (raise ValueError)"""
        a, b = range(int(self.N)), range(int(self.N))
        depth = 3
        if sys.version_info >= (3, 0):
            depth += 1  # curiously enough, Python 3 needs one level more
        self.assertRaises(ValueError, bcolz.eval, "a*3", depth=depth,
                          rootdir=self.rootdir)
        self.assertRaises(ValueError, bcolz.eval, "b*3", depth=depth,
                          rootdir=self.rootdir)

    def test09(self):
        """Testing eval() via expression on __setitem__ (I)"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), b
        c["a + 2 * d - 3 > 0"] = 3
        a[(a + 2 * b - 3) > 0] = 3
        # print "carray ->", c
        # print "numpy  ->", a
        assert_array_equal(c[:], a, "carray[expr] = v does not work correctly")

    def test10(self):
        """Testing eval() via expression on __setitem__ (II)"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), b
        c["a + 2 * d - 3 > 1000"] = 0
        a[(a + 2 * b - 3) > 1000] = 0
        # print "carray ->", c
        # print "numpy  ->", a
        assert_array_equal(c[:], a, "carray[expr] = v does not work correctly")

    def test11(self):
        """Testing eval() with functions like `np.sin()`"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a, rootdir=self.rootdir), bcolz.carray(b)
        if self.vm == "python":
            cr = bcolz.eval("np.sin(c) + 2 * np.log(d) - 3")
        else:
            cr = bcolz.eval("sin(c) + 2 * log(d) - 3")
        nr = np.sin(a) + 2 * np.log(b) - 3
        # print "bcolz.eval ->", cr
        # print "numpy   ->", nr
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test12(self):
        """Testing eval() with `out_flavor` == 'numpy'"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c, d = bcolz.carray(a), bcolz.carray(b, rootdir=self.rootdir)
        cr = bcolz.eval("c + 2 * d - 3", out_flavor='numpy')
        nr = a + 2 * b - 3
        # print "bcolz.eval ->", cr, type(cr)
        # print "numpy   ->", nr
        self.assertTrue(type(cr) == np.ndarray)
        assert_array_equal(cr, nr, "eval does not work correctly")


class evalSmall(evalTest):
    N = 10


class evalDiskSmall(evalTest):
    N = 10
    disk = True


class evalBig(evalTest):
    N = 1e4


class evalDiskBig(evalTest):
    N = 1e4
    disk = True


class evalSmallNE(evalTest):
    N = 10
    vm = "numexpr"


class evalDiskSmallNE(evalTest):
    N = 10
    vm = "numexpr"
    disk = True


class evalBigNE(evalTest):
    N = 1e4
    vm = "numexpr"


class evalDiskBigNE(evalTest):
    N = 1e4
    vm = "numexpr"
    disk = True


class computeMethodsTest(TestCase):

    def test00(self):
        """Testing sum()."""
        a = np.arange(1e5)
        sa = a.sum()
        ac = bcolz.carray(a)
        sac = ac.sum()
        # print "numpy sum-->", sa
        # print "carray sum-->", sac
        self.assertTrue(sa.dtype == sac.dtype,
                        "sum() is not working correctly.")
        self.assertTrue(sa == sac, "sum() is not working correctly.")

    def test01(self):
        """Testing sum() with dtype."""
        a = np.arange(1e5)
        sa = a.sum(dtype='i8')
        ac = bcolz.carray(a)
        sac = ac.sum(dtype='i8')
        # print "numpy sum-->", sa
        # print "carray sum-->", sac
        self.assertTrue(sa.dtype == sac.dtype,
                        "sum() is not working correctly.")
        self.assertTrue(sa == sac, "sum() is not working correctly.")

    def test02(self):
        """Testing sum() with strings (TypeError)."""
        ac = bcolz.zeros(10, 'S3')
        self.assertRaises(TypeError, ac.sum)


class arangeTest(MayBeDiskTest):

    def test00(self):
        """Testing arange() with only a `stop`."""
        a = np.arange(self.N)
        ac = bcolz.arange(self.N, rootdir=self.rootdir)
        self.assertTrue(np.all(a == ac))

    def test01(self):
        """Testing arange() with a `start` and `stop`."""
        a = np.arange(3, self.N)
        ac = bcolz.arange(3, self.N, rootdir=self.rootdir)
        self.assertTrue(np.all(a == ac))

    def test02(self):
        """Testing arange() with a `start`, `stop` and `step`."""
        a = np.arange(3, self.N, 4)
        ac = bcolz.arange(3, self.N, 4, rootdir=self.rootdir)
        self.assertTrue(np.all(a == ac))

    def test03(self):
        """Testing arange() with a `dtype`."""
        a = np.arange(self.N, dtype="i1")
        ac = bcolz.arange(self.N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(np.all(a == ac))


class arange_smallTest(arangeTest, TestCase):
    N = 10
    disk = False


class arange_bigTest(arangeTest, TestCase):
    N = 1e4
    disk = False


class arange_smallDiskTest(arangeTest, TestCase):
    N = 10
    disk = True


class arange_bigDiskTest(arangeTest, TestCase):
    N = 1e4
    disk = True


class constructorTest(MayBeDiskTest):

    def test00(self):
        """Testing carray constructor with an int32 `dtype`."""
        a = np.arange(self.N)
        ac = bcolz.carray(a, dtype='i4', rootdir=self.rootdir)
        self.assertTrue(ac.dtype == np.dtype('i4'))
        a = a.astype('i4')
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test01a(self):
        """Testing zeros() constructor."""
        a = np.zeros(self.N)
        ac = bcolz.zeros(self.N, rootdir=self.rootdir)
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test01b(self):
        """Testing zeros() constructor, with a `dtype`."""
        a = np.zeros(self.N, dtype='i4')
        ac = bcolz.zeros(self.N, dtype='i4', rootdir=self.rootdir)
        # print "dtypes-->", a.dtype, ac.dtype
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test01c(self):
        """Testing zeros() constructor, with a string type."""
        a = np.zeros(self.N, dtype='S5')
        ac = bcolz.zeros(self.N, dtype='S5', rootdir=self.rootdir)
        # print "ac-->", `ac`
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test02a(self):
        """Testing ones() constructor."""
        a = np.ones(self.N)
        ac = bcolz.ones(self.N, rootdir=self.rootdir)
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test02b(self):
        """Testing ones() constructor, with a `dtype`."""
        a = np.ones(self.N, dtype='i4')
        ac = bcolz.ones(self.N, dtype='i4', rootdir=self.rootdir)
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test02c(self):
        """Testing ones() constructor, with a string type"""
        a = np.ones(self.N, dtype='S3')
        ac = bcolz.ones(self.N, dtype='S3', rootdir=self.rootdir)
        # print "a-->", a, ac
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test03a(self):
        """Testing fill() constructor."""
        a = np.ones(self.N)
        ac = bcolz.fill(self.N, 1, rootdir=self.rootdir)
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test03b(self):
        """Testing fill() constructor, with a `dtype`."""
        a = np.ones(self.N, dtype='i4')*3
        ac = bcolz.fill(self.N, 3, dtype='i4', rootdir=self.rootdir)
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))

    def test03c(self):
        """Testing fill() constructor, with a string type"""
        a = np.ones(self.N, dtype='S3')
        ac = bcolz.fill(self.N, "1", dtype='S3', rootdir=self.rootdir)
        # print "a-->", a, ac
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac[:]))


class constructorSmallTest(constructorTest, TestCase):
    N = 10


class constructorSmallDiskTest(constructorTest, TestCase):
    N = 10
    disk = True


class constructorBigTest(constructorTest, TestCase):
    N = 50000


class constructorBigDiskTest(constructorTest, TestCase):
    N = 50000
    disk = True


class dtypesTest(TestCase):

    def test00(self):
        """Testing carray constructor with a float32 `dtype`."""
        a = np.arange(10)
        ac = bcolz.carray(a, dtype='f4')
        self.assertTrue(ac.dtype == np.dtype('f4'))
        a = a.astype('f4')
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac))

    def test01(self):
        """Testing carray constructor with a `dtype` with an empty input."""
        a = np.array([], dtype='i4')
        ac = bcolz.carray([], dtype='f4')
        self.assertTrue(ac.dtype == np.dtype('f4'))
        a = a.astype('f4')
        self.assertTrue(a.dtype == ac.dtype)
        self.assertTrue(np.all(a == ac))

    def test02(self):
        """Testing carray constructor with a plain compound `dtype`."""
        dtype = np.dtype("f4,f8")
        a = np.ones(30000, dtype=dtype)
        ac = bcolz.carray(a, dtype=dtype)
        self.assertTrue(ac.dtype == dtype)
        self.assertTrue(a.dtype == ac.dtype)
        # print "ac-->", `ac`
        assert_array_equal(a, ac[:], "Arrays are not equal")

    def test03(self):
        """Testing carray constructor with a nested compound `dtype`."""
        dtype = np.dtype([('f1', [('f1', 'i2'), ('f2', 'i4')])])
        a = np.ones(3000, dtype=dtype)
        ac = bcolz.carray(a, dtype=dtype)
        self.assertTrue(ac.dtype == dtype)
        self.assertTrue(a.dtype == ac.dtype)
        # print "ac-->", `ac`
        assert_array_equal(a, ac[:], "Arrays are not equal")

    def test04(self):
        """Testing carray constructor with a string `dtype`."""
        a = np.array(["ale", "e", "aco"], dtype="S4")
        ac = bcolz.carray(a, dtype='S4')
        self.assertTrue(ac.dtype == np.dtype('S4'))
        self.assertTrue(a.dtype == ac.dtype)
        # print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test05(self):
        """Testing carray constructor with a unicode `dtype`."""
        a = np.array([u"aŀle", u"eñe", u"açò"], dtype="U4")
        ac = bcolz.carray(a, dtype='U4')
        self.assertTrue(ac.dtype == np.dtype('U4'))
        self.assertTrue(a.dtype == ac.dtype)
        # print "ac-->", `ac`
        assert_array_equal(a, ac, "Arrays are not equal")

    def test06(self):
        """Testing carray constructor with an object `dtype`."""
        dtype = np.dtype("object")
        a = np.array(["ale", "e", "aco"], dtype=dtype)
        ac = bcolz.carray(a, dtype=dtype)
        self.assertEqual(ac.dtype, dtype)
        self.assertEqual(a.dtype, ac.dtype)
        assert_array_equal(a, ac, "Arrays are not equal")

    def test07(self):
        """Checking carray constructor from another carray."""
        types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.float16, np.float32, np.float64,
                 np.complex64, np.complex128]
        if hasattr(np, 'float128'):
            types.extend([np.float128, np.complex256])
        shapes = [(10,), (10, 10), (10, 10, 10)]
        for shape in shapes:
            for t in types:
                a = bcolz.zeros(shape, t)
                b = bcolz.carray(a)
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.shape, shape)


class largeCarrayTest(MayBeDiskTest):

    def test00(self):
        """Creating an extremely large carray (> 2**32) in memory."""

        cn = bcolz.zeros(5e9, dtype="i1")
        self.assertTrue(len(cn) == long(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertTrue(cn[1] == 1)
        cn[int(2e9)] = 2
        self.assertTrue(cn[int(2e9)] == 2)
        cn[long(3e9)] = 3
        self.assertTrue(cn[long(3e9)] == 3)
        cn[-1] = 4
        self.assertTrue(cn[-1] == 4)

        self.assertTrue(cn.sum() == 10)

    def test01(self):
        """Creating an extremely large carray (> 2**32) on disk."""

        cn = bcolz.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == long(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertTrue(cn[1] == 1)
        cn[int(2e9)] = 2
        self.assertTrue(cn[int(2e9)] == 2)
        cn[long(3e9)] = 3
        self.assertTrue(cn[long(3e9)] == 3)
        cn[-1] = 4
        self.assertTrue(cn[-1] == 4)

        self.assertTrue(cn.sum() == 10)

    def test02(self):
        """Opening an extremely large carray (> 2**32) on disk."""

        if not self.disk:
            raise SkipTest
        # Create the array on-disk
        cn = bcolz.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == int(5e9))
        # Reopen it from disk
        cn = bcolz.carray(rootdir=self.rootdir)
        self.assertTrue(len(cn) == int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertTrue(cn[1] == 1)
        cn[int(2e9)] = 2
        self.assertTrue(cn[int(2e9)] == 2)
        cn[long(3e9)] = 3
        self.assertTrue(cn[long(3e9)] == 3)
        cn[-1] = 4
        self.assertTrue(cn[-1] == 4)

        self.assertTrue(cn.sum() == 10)


@skipUnless(is_64bit and common.heavy, "not 64bit or not --heavy")
class largeCarrayMemoryTest(largeCarrayTest, TestCase):
    disk = False


@skipUnless(is_64bit and common.heavy, "not 64bit or not --heavy")
class largeCarrayDiskTest(largeCarrayTest, TestCase):
    disk = True


class persistenceTest(MayBeDiskTest, TestCase):
    disk = True

    def test01a(self):
        """Creating a carray in "r" mode."""

        N = 10000
        self.assertRaises(IOError, bcolz.zeros,
                          N, dtype="i1", rootdir=self.rootdir, mode='r')

    def test01b(self):
        """Creating a carray in "w" mode."""

        N = 50000
        cn = bcolz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == N)

        cn = bcolz.zeros(N-2, dtype="i1", rootdir=self.rootdir, mode='w')
        self.assertTrue(len(cn) == N-2)

        # Now check some accesses (no errors should be raised)
        cn.append([1, 1])
        self.assertTrue(len(cn) == N)
        cn[1] = 2
        self.assertTrue(cn[1] == 2)

    def test01c(self):
        """Creating a carray in "a" mode."""

        N = 30003
        cn = bcolz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == N)

        self.assertRaises(IOError, bcolz.zeros,
                          N-2, dtype="i1", rootdir=self.rootdir, mode='a')

    def test02a(self):
        """Opening a carray in "r" mode."""

        N = 10001
        cn = bcolz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == N)

        cn = bcolz.carray(rootdir=self.rootdir, mode='r')
        self.assertTrue(len(cn) == N)

        # Now check some accesses
        self.assertRaises(IOError, cn.__setitem__, 1, 1)
        self.assertRaises(IOError, cn.append, 1)

    def test02b(self):
        """Opening a carray in "w" mode."""

        N = 100001
        cn = bcolz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == N)

        cn = bcolz.carray(rootdir=self.rootdir, mode='w')
        self.assertTrue(len(cn) == 0)

        # Now check some accesses (no errors should be raised)
        cn.append([1, 1])
        self.assertTrue(len(cn) == 2)
        cn[1] = 2
        self.assertTrue(cn[1] == 2)

    def test02c(self):
        """Opening a carray in "a" mode."""

        N = 1000-1
        cn = bcolz.zeros(N, dtype="i1", rootdir=self.rootdir)
        self.assertTrue(len(cn) == N)

        cn = bcolz.carray(rootdir=self.rootdir, mode='a')
        self.assertTrue(len(cn) == N)

        # Now check some accesses (no errors should be raised)
        cn.append([1, 1])
        self.assertTrue(len(cn) == N+2)
        cn[1] = 2
        self.assertTrue(cn[1] == 2)
        cn[N+1] = 3
        self.assertTrue(cn[N+1] == 3)


class bloscCompressorsTest(MayBeDiskTest, TestCase):

    def tearDown(self):
        # Restore defaults
        bcolz.cparams.setdefaults(clevel=5, shuffle=True, cname='blosclz')
        MayBeDiskTest.tearDown(self)

    def test00(self):
        """Testing all available compressors in small arrays"""
        a = np.arange(20)
        cnames = bcolz.blosc_compressor_list()
        if common.verbose:
            print("Checking compressors:", cnames)
        # print "\nsize b uncompressed-->", a.size * a.dtype.itemsize
        for cname in cnames:
            b = bcolz.carray(a, rootdir=self.rootdir,
                             cparams=bcolz.cparams(clevel=9, cname=cname))
            # print "size b compressed  -->", b.cbytes, "with '%s'"%cname
            self.assertTrue(sys.getsizeof(b) > b.nbytes,
                            "compression does not seem to have any overhead")
            assert_array_equal(a, b[:], "Arrays are not equal")
            # Remove the array on disk before trying with the next one
            if self.disk:
                common.remove_tree(self.rootdir)

    def test01a(self):
        """Testing all available compressors in big arrays (setdefaults)"""
        a = np.arange(1e5)
        cnames = bcolz.blosc_compressor_list()
        if common.verbose:
            print("Checking compressors:", cnames)
        # print "\nsize b uncompressed-->", a.size * a.dtype.itemsize
        for cname in cnames:
            bcolz.cparams.setdefaults(clevel=9, cname=cname)
            b = bcolz.carray(a, rootdir=self.rootdir)
            # print "size b compressed  -->", b.cbytes, "with '%s'"%cname
            self.assertTrue(sys.getsizeof(b) < b.nbytes,
                            "carray does not seem to compress at all")
            assert_array_equal(a, b[:], "Arrays are not equal")
            # Remove the array on disk before trying with the next one
            if self.disk:
                common.remove_tree(self.rootdir)

    def test01b(self):
        """Testing all available compressors in big arrays (bcolz.defaults)"""
        a = np.arange(1e5)
        cnames = bcolz.blosc_compressor_list()
        if common.verbose:
            print("Checking compressors:", cnames)
        # print "\nsize b uncompressed-->", a.size * a.dtype.itemsize
        for cname in cnames:
            bcolz.defaults.cparams = {
                'clevel': 9, 'shuffle': True, 'cname': cname}
            b = bcolz.carray(a, rootdir=self.rootdir)
            # print "size b compressed  -->", b.cbytes, "with '%s'"%cname
            self.assertTrue(sys.getsizeof(b) < b.nbytes,
                            "carray does not seem to compress at all")
            assert_array_equal(a, b[:], "Arrays are not equal")
            # Remove the array on disk before trying with the next one
            if self.disk:
                common.remove_tree(self.rootdir)
        # Restore defaults
        bcolz.defaults.cparams = {
            'clevel': 5, 'shuffle': True, 'cname': 'blosclz'}


class compressorsMemoryTest(bloscCompressorsTest, TestCase):
    disk = False


class compressorsDiskTest(bloscCompressorsTest, TestCase):
    disk = True


class reprTest(TestCase):
    def test_datetime_carray_day(self):
        ct = carray(np.array(['2010-01-01', '2010-01-02'],
                             dtype='datetime64[D]'))
        result = repr(ct)
        self.assertTrue("['2010-01-01' '2010-01-02']" in result)

    def test_datetime_carray_nanos(self):
        x = ['2014-12-29T17:57:59.000000123',
             '2014-12-29T17:57:59.000000456']
        ct = carray(np.array(x, dtype='datetime64[ns]'))
        result = repr(ct)
        for el in x:
            self.assertTrue(el in result)

class PurgeDiskArrayTest(MayBeDiskTest, TestCase):
    disk = True

    def test_purge(self):
        b = bcolz.arange(1e2, rootdir=self.rootdir)
        b.purge()
        self.assertFalse(os.path.isdir(self.rootdir))

    def test_purge_fails_for_missing_directory(self):
        b = bcolz.arange(1e2, rootdir=self.rootdir)
        shutil.rmtree(self.rootdir)
        # OSError is fairly un-specific, but better than nothing
        self.assertRaises(OSError, b.purge)

class PurgeMemoryArrayTest(MayBeDiskTest, TestCase):
    disk = False

    def test_purge(self):
        b = bcolz.arange(1e2, rootdir=self.rootdir)
        # this should work and should be a noop
        b.purge()

class reprDiskTest(MayBeDiskTest,TestCase):
    disk = True

    def _create_expected(self, mode):
        expected = textwrap.dedent("""
                   carray((0,), float64)
                     nbytes: 0; cbytes: 16.00 KB; ratio: 0.00
                     cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
                     rootdir := '%s'
                     mode    := '%s'
                   []
                   """ % (self.rootdir, mode)).strip()
        return expected


    def test_repr_disk_array_write(self):
        x = carray([], rootdir=self.rootdir, mode='w')
        expected = self._create_expected('w')
        self.assertEqual(expected, repr(x))

    def test_repr_disk_array_read(self):
        x = carray([], rootdir=self.rootdir, mode='r')
        expected = self._create_expected('r')
        self.assertEqual(expected, repr(x))

    def test_repr_disk_array_append(self):
        x = carray([], rootdir=self.rootdir, mode='w')
        y = carray(rootdir=self.rootdir)
        expected = self._create_expected('a')
        self.assertEqual(expected, repr(y))

class chunksIterTest(MayBeDiskTest):
    def test00(self):
        """Testing chunk iterator"""
        _dtype = 'int64'
        chunklen_ = 10
        N = 1e2 + 3
        a = np.arange(N, dtype=_dtype)
        b = bcolz.carray(a, dtype=_dtype, chunklen=chunklen_,
                         rootdir=self.rootdir)
        # print 'nchunks', b.nchunks
        for n, chunk_ in enumerate(b.chunks):
            # print 'chunk nr.', n, '-->', chunk_
            assert_array_equal(chunk_[0:chunklen_],
                               a[n * chunklen_:(n + 1) * chunklen_],
                               "iter chunks not working correctly")

        self.assertEquals(n, len(a) // chunklen_ - 1)


class chunksIterMemoryTest(chunksIterTest, TestCase):
    disk = False


class chunksIterDiskTest(chunksIterTest, TestCase):
    disk = True

class ContextManagerTest(MayBeDiskTest, TestCase):
    disk = True

    def test_with_statement_flushes(self):

        with carray([], rootdir=self.rootdir, mode='w') as x:
            x.append(1)
        received = np.array(carray(rootdir=self.rootdir))
        expected = np.array([1])
        assert_array_equal(expected, received)

    def test_with_read_only(self):
        x = bcolz.arange(5, rootdir=self.rootdir, mode="w")
        x.flush()
        sx = sum(i for i in x)

        with bcolz.open(self.rootdir, mode='r') as xreadonly:
            sxreadonly = sum(i for i in xreadonly)
        self.assertEquals(sx, sxreadonly)


class nleftoversTest(TestCase):

    def test_empty(self):
        a = carray([])
        self.assertEquals(0, a.nleftover)

    def test_one(self):
        a = carray([1])
        self.assertEqual(1, a.nleftover)

    def test_beyond_one(self):
        a = carray(np.zeros(2049), chunklen=2048)
        self.assertEqual(1, a.nleftover)


class LeftoverTest(MayBeDiskTest):

    def test_leftover_ptr(self):
        typesize = 8
        items = 7
        a = carray([i for i in range(items)], dtype='i8', rootdir=self.rootdir)
        for i in range(items):
            out = ctypes.c_int64.from_address(a.leftover_ptr + (i * typesize))
            self.assertEqual(i, out.value)

    def test_leftover_ptr_after_chunks(self):
        typesize = 4
        items = 108
        chunklen = 100
        a = carray([i for i in range(items)], chunklen=chunklen, dtype='i4',
                   rootdir=self.rootdir)
        for i in range(items % chunklen):
            out = ctypes.c_int32.from_address(a.leftover_ptr + (i * typesize))
            self.assertEqual(chunklen + i, out.value)

    def test_leftover_array(self):
        items = 7
        a = carray([i for i in range(items)], dtype='i4', rootdir=self.rootdir)
        reference = np.array([0, 1, 2, 3, 4, 5, 6], dtype='int32')
        out = a.leftover_array[:items]
        assert_array_equal(reference, out)

    def test_leftover_bytes(self):
        typesize = 8
        items = 9
        a = carray([i for i in range(items)], dtype='i8', rootdir=self.rootdir)
        self.assertEqual(a.leftover_bytes, items * typesize)

    def test_leftover_elements(self):
        items = 9
        a = carray([i for i in range(items)], dtype='i8', rootdir=self.rootdir)
        self.assertEqual(a.leftover_elements, items)


class LeftoverMemoryTest(LeftoverTest, TestCase):
    disk = False


class LeftoverDiskTest(LeftoverTest, TestCase):
    disk = True

    def test_leftover_ptr_create_flush_open(self):
        typesize = 8
        items = 120
        chunklen = 50
        n_leftovers = items % chunklen
        n_chunks = items // chunklen

        a = carray([i for i in range(items)], chunklen=chunklen, dtype='i8',
                   rootdir=self.rootdir)
        a.flush()

        b = carray(rootdir=self.rootdir)
        for i in range(n_leftovers):
            out = ctypes.c_int32.from_address(b.leftover_ptr + (i * typesize))
            self.assertEqual((n_chunks * chunklen) + i, out.value)

    def test_leftover_ptr_with_statement_create_open(self):
        typesize = 8
        items = 120
        chunklen = 50
        n_leftovers = items % chunklen
        n_chunks = items // chunklen

        ca = carray([], chunklen=chunklen, dtype='i8', rootdir=self.rootdir)
        with ca as a:
            for i in range(items):
                a.append(i)

        b = carray(rootdir=self.rootdir)
        for i in range(n_leftovers):
            out = ctypes.c_int32.from_address(b.leftover_ptr + (i * typesize))
            self.assertEqual((n_chunks * chunklen) + i, out.value)

    def test_repr_of_empty_object_array(self):
        assert 'ratio: nan' in repr(carray(np.array([], dtype=object)))


class MagicNumbers(MayBeDiskTest):

    def test_type_i4(self):
        N = 2**16
        ca = carray([i for i in range(N)], dtype='i4', rootdir=self.rootdir)

        for i in range(len(ca)):
            v = ca[i]
            self.assertTrue(isinstance(v, _inttypes))

    def test_type_i8(self):
        N = 2**15
        ca = carray([i for i in range(N)], dtype='i8', rootdir=self.rootdir)

        for i in range(len(ca)):
            v = ca[i]
            self.assertTrue(isinstance(v, _inttypes))

    def test_type_f8(self):
        N = 2**15
        ca = carray([i for i in range(N)], dtype='f8', rootdir=self.rootdir)

        for i in range(len(ca)):
            v = ca[i]
            self.assertTrue(isinstance(v, float))


class MagicNumbersMemoryTest(MagicNumbers, TestCase):
    disk = False


class MagicNumbersDiskTest(MagicNumbers, TestCase):
    disk = True


if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# tab-width: 4
# fill-column: 72
# End:
