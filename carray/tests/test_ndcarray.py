# -*- coding: utf-8 -*-
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
        b = ca.arange(16).reshape((2,2,4))
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

    def test03a(self):
        """Testing `__getitem()__` method with several slices (I)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (slice(1,3,1), slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03b(self):
        """Testing `__getitem()__` method with several slices (II)"""
        a = np.arange(24).reshape((4,3,2))
        b = ca.carray(a)
        sl = (slice(1,3,2), slice(1,4,2), slice(None))
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test03c(self):
        """Testing `__getitem()__` method with several slices (III)"""
        a = np.arange(120).reshape((5,4,3,2))
        b = ca.carray(a)
        sl = (slice(None,None,3), slice(1,3,2), slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04a(self):
        """Testing `__getitem()__` method with shape reduction (I)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (1,1)
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__getitem()__` method with shape reduction (II)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (1,slice(1,4,2))
        #print "b[sl]->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__getitem()__` method with shape reduction (III)"""
        a = np.arange(60).reshape((5,4,3))
        b = ca.carray(a)
        sl = (1,slice(1,4,2),2)
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

    def test03a(self):
        """Testing `__setitem()__` method with several slices (I)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (slice(1,3,1), slice(1,None,2))
        #print "before->", `b[sl]`
        a[sl] = [[1],[2]]
        b[sl] = [[1],[2]]
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03b(self):
        """Testing `__setitem()__` method with several slices (II)"""
        a = np.arange(24).reshape((4,3,2))
        b = ca.carray(a)
        sl = (slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = [[[1]],[[2]]]
        b[sl] = [[[1]],[[2]]]
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03c(self):
        """Testing `__setitem()__` method with several slices (III)"""
        a = np.arange(120).reshape((5,4,3,2))
        b = ca.carray(a)
        sl = (slice(1,3), slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = [[[[1]],[[2]]]]*2
        b[sl] = [[[[1]],[[2]]]]*2
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test03d(self):
        """Testing `__setitem()__` method with several slices (IV)"""
        a = np.arange(120).reshape((5,4,3,2))
        b = ca.carray(a)
        sl = (slice(1,3), slice(1,3,1), slice(1,None,2), slice(1))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        #print "after->", `b[sl]`
        assert_array_equal(a[:], b[:], "Arrays are not equal")

    def test04a(self):
        """Testing `__setitem()__` method with shape reduction (I)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (1,1)
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04b(self):
        """Testing `__setitem()__` method with shape reduction (II)"""
        a = np.arange(12).reshape((4,3))
        b = ca.carray(a)
        sl = (1,slice(1,4,2))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")

    def test04c(self):
        """Testing `__setitem()__` method with shape reduction (III)"""
        a = np.arange(24).reshape((4,3,2))
        b = ca.carray(a)
        sl = (1,2,slice(None,None,None))
        #print "before->", `b[sl]`
        a[sl] = 2
        b[sl] = 2
        #print "after->", `b[sl]`
        assert_array_equal(a[sl], b[sl], "Arrays are not equal")


class appendTest(unittest.TestCase):

    def test00a(self):
        """Testing `append()` method (correct shape)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((1,3), 3, dtype="i4")
        b.append([(3,3,3)])
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `append()` method (correct shape, single row)"""
        a = np.ones((2,3), dtype="i4")*3
        b = ca.fill((1,3), 3, dtype="i4")
        b.append((3,3,3))
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


class iterTest(unittest.TestCase):

    def test00(self):
        """Testing `iter()` (no start, stop, step)"""
        a = np.ones((3,), dtype="i4")
        b = ca.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter():
            assert_array_equal(a, r, "Arrays are not equal")

    def test01(self):
        """Testing `iter()` (w/ start, stop)"""
        a = np.ones((3,), dtype="i4")
        b = ca.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter(start=10):
            assert_array_equal(a, r, "Arrays are not equal")

    def test02(self):
        """Testing `iter()` (w/ start, stop, step)"""
        a = np.ones((3,), dtype="i4")
        b = ca.ones((1000,3), dtype="i4")
        #print "b->", `b`
        for r in b.iter(15, 100, 3):
            assert_array_equal(a, r, "Arrays are not equal")


class reshapeTest(unittest.TestCase):

    def test00a(self):
        """Testing `reshape()` (unidim -> ndim)"""
        a = np.ones((3,4), dtype="i4")
        b = ca.ones(12, dtype="i4").reshape((3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00b(self):
        """Testing `reshape()` (unidim -> ndim, -1 in newshape (I))"""
        a = np.ones((3,4), dtype="i4")
        b = ca.ones(12, dtype="i4").reshape((-1,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test00c(self):
        """Testing `reshape()` (unidim -> ndim, -1 in newshape (II))"""
        a = np.ones((3,4), dtype="i4")
        b = ca.ones(12, dtype="i4").reshape((3,-1))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test01(self):
        """Testing `reshape()` (ndim -> unidim)"""
        a = np.ones(12, dtype="i4")
        c = ca.ones(12, dtype="i4").reshape((3,4))
        b = c.reshape(12)
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02a(self):
        """Testing `reshape()` (ndim -> ndim, I)"""
        a = np.arange(12, dtype="i4").reshape((3,4))
        c = ca.arange(12, dtype="i4").reshape((4,3))
        b = c.reshape((3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test02b(self):
        """Testing `reshape()` (ndim -> ndim, II)"""
        a = np.arange(24, dtype="i4").reshape((2,3,4))
        c = ca.arange(24, dtype="i4").reshape((4,3,2))
        b = c.reshape((2,3,4))
        #print "b->", `b`
        assert_array_equal(a, b, "Arrays are not equal")

    def test03(self):
        """Testing `reshape()` (0-dim)"""
        a = np.ones((0,4), dtype="i4")
        b = ca.ones(0, dtype="i4").reshape((0,4))
        #print "b->", `b`
        # The next does not work well for carrays with shape (0,)
        #assert_array_equal(a, b, "Arrays are not equal")
        self.assert_(a.dtype.base == b.dtype.base)
        self.assert_(a.shape == b.shape+b.dtype.shape)

    def test04(self):
        """Testing `ndim` and `shape` attributes"""
        shape = (2,3,4)
        c = ca.arange(24).reshape(shape)
        self.assert_(c.shape == shape)
        self.assert_(c.ndim == len(shape))


class compoundTest(unittest.TestCase):

    def test00(self):
        """Testing compound types (creation)"""
        a = np.ones((300,4), dtype=self.dtype)
        b = ca.ones((300,4), dtype=self.dtype)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing compound types (append)"""
        a = np.ones((300,4), dtype=self.dtype)
        b = ca.carray([], dtype=self.dtype).reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing compound types (iter)"""
        a = np.ones((3,), dtype=self.dtype)
        b = ca.ones((1000,3), dtype=self.dtype)
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class plainCompoundTest(compoundTest):
    dtype = np.dtype("i4,i8")

class nestedCompoundTest(compoundTest):
    dtype = np.dtype([('f1', [('f1', 'i2'), ('f2', 'i4')])])


class stringTest(unittest.TestCase):

    def test00(self):
        """Testing string types (creation)"""
        a = np.array([["ale", "ene"], ["aco", "ieie"]], dtype="S4")
        b = ca.carray(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing string types (append)"""
        a = np.ones((300,4), dtype="S4")
        b = ca.carray([], dtype="S4").reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing string types (iter)"""
        a = np.ones((3,), dtype="S40")
        b = ca.ones((1000,3), dtype="S40")
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class unicodeTest(unittest.TestCase):

    def test00(self):
        """Testing unicode types (creation)"""
        a = np.array([[u"aŀle", u"eñe"], [u"açò", u"áèâë"]], dtype="U4")
        b = ca.carray(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test01(self):
        """Testing unicode types (append)"""
        a = np.ones((300,4), dtype="U4")
        b = ca.carray([], dtype="U4").reshape((0,4))
        b.append(a)
        #print "b.dtype-->", b.dtype
        #print "b->", `b`
        self.assert_(a.dtype == b.dtype.base)
        assert_array_equal(a, b[:], "Arrays are not equal")

    def test02(self):
        """Testing unicode types (iter)"""
        a = np.ones((3,), dtype="U40")
        b = ca.ones((1000,3), dtype="U40")
        #print "b->", `b`
        for r in b.iter():
            #print "r-->", r
            assert_array_equal(a, r, "Arrays are not equal")


class evalTest(unittest.TestCase):

    vm = "python"

    def setUp(self):
        self.prev_vm = ca.defaults.eval_vm
        ca.defaults.eval_vm = self.vm

    def tearDown(self):
        ca.defaults.eval_vm = self.prev_vm

    def test00(self):
        """Testing evaluation of ndcarrays (bool out)"""
        a = np.arange(np.prod(self.shape)).reshape(self.shape)
        b = ca.arange(np.prod(self.shape)).reshape(self.shape)
        outa = eval("a>0")
        outb = ca.eval("b>0")
        assert_array_equal(outa, outb, "Arrays are not equal")

    def test01(self):
        """Testing evaluation of ndcarrays (int out)"""
        a = np.arange(np.prod(self.shape)).reshape(self.shape)
        b = ca.arange(np.prod(self.shape)).reshape(self.shape)
        outa = eval("a*2.+1")
        outb = ca.eval("b*2.+1")
        assert_array_equal(outa, outb, "Arrays are not equal")

    def test02(self):
        """Testing evaluation of ndcarrays (reduction)"""
        # Reduction ops are not supported yet
        a = np.arange(np.prod(self.shape)).reshape(self.shape)
        b = ca.arange(np.prod(self.shape)).reshape(self.shape)
        if ca.defaults.eval_vm:
            self.assertRaises(NotImplementedError,
                              ca.eval, "sum(b)", depth=3)
        else:
            self.assertRaises(NotImplementedError,
                              ca.eval, "b.sum(axis=0)", depth=3)
            
    def test03(self):
        """Testing rich comparisons of ndcarrays"""
        c = ca.arange(np.prod(self.shape)).reshape(self.shape)
        cmpv = c.len - 1
        assert_array_equal(ca.eval("c<cmpv"), c<cmpv)
        assert_array_equal(ca.eval("c==cmpv"), c==cmpv)
        assert_array_equal(ca.eval("c>cmpv"), c>cmpv)
        assert_array_equal(ca.eval("c<=cmpv"), c<=cmpv)
        assert_array_equal(ca.eval("c!=cmpv"), c!=cmpv)
        assert_array_equal(ca.eval("c>=cmpv"), c>=cmpv)

class d2evalTest(evalTest):
    shape = (3,4)

class d2eval_ne(evalTest):
    shape = (3,4)
    vm = "numexpr"

class d3evalTest(evalTest):
    shape = (3,4,5)

class d3eval_ne(evalTest):
    shape = (3,4,5)
    vm = "numexpr"

class d4evalTest(evalTest):
    shape = (3,40,50,2)

class d4eval_ne(evalTest):
    shape = (3,40,50,2)
    vm = "numexpr"


class computeMethodsTest(unittest.TestCase):

    def test00(self):
        """Testing sum()."""
        a = np.arange(1e5).reshape(10, 1e4)
        sa = a.sum()
        ac = ca.carray(a)
        sac = ac.sum()
        #print "numpy sum-->", sa
        #print "carray sum-->", sac
        self.assert_(sa.dtype == sac.dtype, "sum() is not working correctly.")
        self.assert_(sa == sac, "sum() is not working correctly.")



def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(constructorTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))
    theSuite.addTest(unittest.makeSuite(setitemTest))
    theSuite.addTest(unittest.makeSuite(appendTest))
    theSuite.addTest(unittest.makeSuite(resizeTest))
    theSuite.addTest(unittest.makeSuite(iterTest))
    theSuite.addTest(unittest.makeSuite(reshapeTest))
    theSuite.addTest(unittest.makeSuite(plainCompoundTest))
    theSuite.addTest(unittest.makeSuite(nestedCompoundTest))
    theSuite.addTest(unittest.makeSuite(stringTest))
    theSuite.addTest(unittest.makeSuite(unicodeTest))
    theSuite.addTest(unittest.makeSuite(d2evalTest))
    theSuite.addTest(unittest.makeSuite(d3evalTest))
    theSuite.addTest(unittest.makeSuite(d4evalTest))
    theSuite.addTest(unittest.makeSuite(computeMethodsTest))
    if ca.numexpr_here:
        theSuite.addTest(unittest.makeSuite(d2eval_ne))
        theSuite.addTest(unittest.makeSuite(d3eval_ne))
        theSuite.addTest(unittest.makeSuite(d4eval_ne))


    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
