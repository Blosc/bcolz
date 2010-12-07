########################################################################
#
#       License: BSD
#       Created: September 1, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
import unittest


class createTest(unittest.TestCase):

    def test00(self):
        """Testing ctable creation from a tuple of carrays"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'))
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing ctable creation from a tuple of numpy arrays"""
        N = 1e1
        a = np.arange(N, dtype='i4')
        b = np.arange(N, dtype='f8')+1
        t = ca.ctable((a, b), ('f0', 'f1'))
        #print "t->", `t`
        ra = np.rec.fromarrays([a,b]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing ctable creation from an structured array"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing ctable creation from large iterator"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing ctable creation from large iterator (with a hint)"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)),
                         dtype='i4,f8', count=N)
        t = ca.fromiter(((i, i*2.) for i in xrange(N)),
                        dtype='i4,f8', count=N)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")


class add_del_colTest(unittest.TestCase):

    def test00(self):
        """Testing adding a new column (carray flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(ca.carray(c), 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01b(self):
        """Testing cparams when adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, cparams=ca.cparams(1))
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        self.assert_(t['f2'].cparams.clevel == 1, "Incorrect clevel")

    def test02(self):
        """Testing adding a new column (default naming)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(ca.carray(c))
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing inserting a new column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, name='c0', pos=0)
        ra = np.fromiter(((i*3, i, i*2.) for i in xrange(N)), dtype='i8,i4,f8')
        ra.dtype.names = ('c0', 'f0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing inserting a new column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, name='c0', pos=1)
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        ra.dtype.names = ('f0', 'c0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test05(self):
        """Testing removing an existing column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra)
        t.delcol(pos=0)
        # The next gives a segfault.  See:
        # http://projects.scipy.org/numpy/ticket/1598
        #ra = np.fromiter(((i*3, i*2) for i in xrange(N)), dtype='i8,f8')
        #ra.dtype.names = ('f1', 'f2')
        dt = np.dtype([('f1', 'i8'), ('f2', 'f8')])
        ra = np.fromiter(((i*3, i*2) for i in xrange(N)), dtype=dt)
        #print "t->", `t`
        #print "ra", ra
        #assert_array_equal(t[:], ra, "ctable values are not correct")

    def test06(self):
        """Testing removing an existing column (at the end)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra)
        t.delcol(pos=2)
        ra = np.fromiter(((i, i*3) for i in xrange(N)), dtype='i4,i8')
        ra.dtype.names = ('f0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test07(self):
        """Testing removing an existing column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra)
        t.delcol(pos=1)
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test08(self):
        """Testing removing an existing column (by name)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra)
        t.delcol('f1')
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")


class getitemTest(unittest.TestCase):

    def test00(self):
        """Testing __getitem__ with only a start"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        start = 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start], ra[start], "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with start, stop"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        start, stop = 3, 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start:stop], ra[start:stop],
                           "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with start, stop, step"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        start, stop, step = 3, 9, 2
        #print "t->", `t[start:stop:step]`
        #print "ra->", ra[start:stop:step]
        assert_array_equal(t[start:stop:step], ra[start:stop:step],
                           "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with a column name"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        colname = "f1"
        #print "t->", `t[colname]`
        #print "ra->", ra[colname]
        assert_array_equal(t[colname][:], ra[colname],
                           "ctable values are not correct")

    def test04(self):
        """Testing __getitem__ with a list of column names"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        colnames = ["f0", "f2"]
        #print "t->", `t[colnames]`
        #print "ra->", ra[colnames]
        assert_array_equal(t[colnames][:], ra[colnames],
                           "ctable values are not correct")


class setitemTest(unittest.TestCase):

    def test00(self):
        """Testing __setitem__ with only a start"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10)
        sl = slice(9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing __setitem__ with only a stop"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10)
        sl = slice(None, 9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing __setitem__ with a start, stop"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10)
        sl = slice(1,90, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing __setitem__ with a start, stop, step"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10)
        sl = slice(1,90, 2)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing __setitem__ with a large step"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10)
        sl = slice(1,43, 20)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")



class appendTest(unittest.TestCase):

    def test00(self):
        """Testing append() with scalar values"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        t.append((N, N*2))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+1)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing append() with numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((a, b))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing append() with carrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((ca.carray(a), ca.carray(b)))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing append() with structured arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t.append(ra2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing append() with another ctable"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t2 = ca.ctable(ra2)
        t.append(t2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")


class copyTest(unittest.TestCase):

    def test00(self):
        """Testing copy() without params"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        t2 = t.copy()
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t2.append((a, b))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        self.assert_(len(t) == N, "copy() does not work correctly")
        self.assert_(len(t2) == N+10, "copy() does not work correctly")
        assert_array_equal(t2[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing copy() with higher clevel"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        t2 = t.copy(cparams=ca.cparams(clevel=9))
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t.cparams.clevel == ca.cparams().clevel)
        self.assert_(t2.cparams.clevel == 9)
        self.assert_(t['f1'].cbytes > t2['f1'].cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lower clevel"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        t2 = t.copy(cparams=ca.cparams(clevel=1))
        self.assert_(t.cparams.clevel == ca.cparams().clevel)
        self.assert_(t2.cparams.clevel == 1)
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")

    def test03(self):
        """Testing copy() with no shuffle"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        t2 = t.copy(cparams=ca.cparams(shuffle=False))
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")


class specialTest(unittest.TestCase):

    def test00(self):
        """Testing __len__()"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        self.assert_(len(t) == len(ra), "Objects do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big ctables)"""
        N = int(1e4)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        #print "size t uncompressed ->", t.nbytes
        #print "size t compressed   ->", t.cbytes
        self.assert_(sys.getsizeof(t) < t.nbytes,
                     "ctable does not seem to compress at all")

    def test02(self):
        """Testing __sizeof__() (small ctables)"""
        N = int(111)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        #print "size t uncompressed ->", t.nbytes
        #print "size t compressed   ->", t.cbytes
        self.assert_(sys.getsizeof(t) > t.nbytes,
                     "ctable compress too much??")


class evalTest(unittest.TestCase):

    def test00(self):
        """Testing eval() with only columns"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        ctr = t.eval("f0 * f1 * f2")
        rar = ra['f0'] * ra['f1'] * ra['f2']
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test01(self):
        """Testing eval() with columns and constants"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        ctr = t.eval("f0 * f1 * 3")
        rar = ra['f0'] * ra['f1'] * 3
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test02(self):
        """Testing eval() with columns, constants and other variables"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        var_ = 10.
        ctr = t.eval("f0 * f2 * var_")
        rar = ra['f0'] * ra['f2'] * var_
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test03(self):
        """Testing eval() with columns and numexpr functions"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        ctr = t.eval("f0 * sin(f1)")
        rar = ra['f0'] * np.sin(ra['f1'])
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_almost_equal(ctr[:], rar, decimal=15,
                                  err_msg="ctable values are not correct")

    def test04(self):
        """Testing eval() with a boolean as output"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        ctr = t.eval("f0 >= f1")
        rar = ra['f0'] >= ra['f1']
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test05(self):
        """Testing eval() with a mix of columns and numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        a = np.arange(N)
        b = np.arange(N)
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test06(self):
        """Testing eval() with a mix of columns, numpy arrays and lists"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        a = np.arange(N)
        b = np.arange(N).tolist()
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")


class fancy_indexing_getitemTest(unittest.TestCase):

    def test00(self):
        """Testing fancy indexing with a small list"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[3,1]]
        rar = ra[[3,1]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing with a large numpy array"""
        N = 10*1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        idx = np.random.randint(1000, size=1000)
        rt = t[idx]
        rar = ra[idx]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing fancy indexing with an empty list"""
        N = 10*1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[]]
        rar = ra[[]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        N = 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[2.3, 5.6]]
        rar = ra[[2.3, 5.6]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1,101)
        b = ca.carray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(IndexError, b.__getitem__, idx)


class fancy_indexing_setitemTest(unittest.TestCase):

    def test00a(self):
        """Testing fancy indexing (setitem) with a small list"""
        N = 100
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = [3,1]
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing fancy indexing (setitem) with a small list (II)"""
        N = 100
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = [3,1]
        t[sl] = [(-1, -2, -3), (-3, -2, -1)]
        ra[sl] = [(-1, -2, -3), (-3, -2, -1)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing (setitem) with a large array"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(N, size=100)
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02a(self):
        """Testing fancy indexing (setitem) with a boolean array (I)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(2, size=1000).astype('bool')
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02b(self):
        """Testing fancy indexing (setitem) with a boolean array"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(10, size=1000).astype('bool')
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing fancy indexing (setitem) with a boolean array (all false)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.zeros(N, dtype="bool")
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing fancy indexing (setitem) with a boolean array (all true)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.ones(N, dtype="bool")
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")


class iterTest(unittest.TestCase):

    def test00(self):
        """Testing ctable.__iter__"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4)
        cl = [r['f1'] for r in t]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test01(self):
        """Testing ctable.iter() without params"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4)
        cl = [r['f1'] for r in t.iter()]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test02(self):
        """Testing ctable.iter() with start,stop,step"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4)
        cl = [r['f1'] for r in t.iter(1,9,3)]
        nl = [r['f1'] for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test03(self):
        """Testing ctable.iter() with outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4)
        cl = [tuple(r) for r in t.iter(outcols=['f2', '__nrow__', 'f0'])]
        nl = [(r['f2'], i, r['f0']) for i, r in enumerate(ra)]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test04(self):
        """Testing ctable.iter() with start,stop,step and outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4)
        cl = [tuple(r) for r in t.iter(1,9,3, ['f2', '__nrow__', 'f0'])]
        nl = [(r['f2'], r['f0'], r['f0']) for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")


class eval_getitemTest(unittest.TestCase):

    def test00(self):
        """Testing __getitem__ with an expression (all false values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t['f1 > f2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i > i*2.),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with an expression (all true values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t['f1 <= f2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i <= i*2.),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with an expression (true/false values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t['f1*4 >= f2*2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with an invalid expression"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        # In t['f1*4 >= ppp'], 'ppp' variable name should be found
        self.assertRaises(NameError, t.__getitem__, 'f1*4 >= ppp')

    def test04a(self):
        """Testing __getitem__ with an expression with columns and ndarrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        c2 = t['f2'][:]
        rt = t['f1*4 >= c2*2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test04b(self):
        """Testing __getitem__ with an expression with columns and carrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        c2 = t['f2']
        rt = t['f1*4 >= c2*2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test05(self):
        """Testing __getitem__ with an expression with overwritten vars"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        f1 = t['f2']
        f2 = t['f1']
        rt = t['f2*4 >= f1*2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")


class bool_getitemTest(unittest.TestCase):

    def test00(self):
        """Testing __getitem__ with a boolean array (all false values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = t.eval('f1 > f2')
        rt = t[barr]
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i > i*2.),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with a boolean array (mixed values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = t.eval('f1*4 >= f2*2')
        rt = t[barr]
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with a short boolean array"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = np.zeros(len(t)-1, dtype=np.bool_)
        self.assertRaises(IndexError, t.__getitem__, barr)


class getifTest(unittest.TestCase):

    def test00a(self):
        """Testing getif() with a boolean array (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = t.eval('f1 > f2')
        rt = [r['f0'] for r in t.getif(barr)]
        rl = [i for i in xrange(N) if i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test00b(self):
        """Testing getif() with a boolean array (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = t.eval('f1 <= f2')
        rt = [r['f0'] for r in t.getif(barr)]
        rl = [i for i in xrange(N) if i <= i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test00c(self):
        """Testing getif() with a boolean array (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        barr = t.eval('4+f1 > f2')
        rt = [r['f0'] for r in t.getif(barr)]
        rl = [i for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test01a(self):
        """Testing getif() with an expression (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [r['f0'] for r in t.getif('f1 > f2')]
        rl = [i for i in xrange(N) if i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test01b(self):
        """Testing getif() with an expression (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [r['f0'] for r in t.getif('f1 <= f2')]
        rl = [i for i in xrange(N) if i <= i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test01c(self):
        """Testing getif() with an expression (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [r['f0'] for r in t.getif('4+f1 > f2')]
        rl = [i for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test02a(self):
        """Testing getif() with an expression (with outcols)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [r['f1'] for r in t.getif('4+f1 > f2', outcols=['f1'])]
        rl = [i*2. for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test02b(self):
        """Testing getif() with an expression (with outcols II)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [(r['f1'], r['f2']) for r in t.getif('4+f1 > f2',
                                                 outcols=['f1','f2'])]
        rl = [(i*2., i*3) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    def test02c(self):
        """Testing getif() with an expression (with outcols III)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [(r['f2'], r['f0']) for r in t.getif('4+f1 > f2',
                                                 outcols=['f2','f0'])]
        rl = [(i*3, i) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "getif not working correctly")

    # This does not work anymore because of the nesting of ctable._iter
    def _test02d(self):
        """Testing getif() with an expression (with outcols IV)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        getif = t.getif('f1 > f2', outcols=['f3','f0'])
        self.assertRaises(ValueError, getif.next)

    def test03(self):
        """Testing getif() with an expression (with __nrow__ in outcols)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = [tuple(r) for r in t.getif('4+f1 > f2',
                                        outcols=['__nrow__','f2','f0'])]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt, type(rt[0][0])
        #print "rl->", rl, type(rl[0][0])
        self.assert_(rt == rl, "getif not working correctly")


class getif_smallTest(getifTest):
    N = 10

class getif_largeTest(getifTest):
    N = 10*1000


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(createTest))
    theSuite.addTest(unittest.makeSuite(add_del_colTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))
    theSuite.addTest(unittest.makeSuite(setitemTest))
    theSuite.addTest(unittest.makeSuite(appendTest))
    theSuite.addTest(unittest.makeSuite(copyTest))
    theSuite.addTest(unittest.makeSuite(specialTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_getitemTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_setitemTest))
    theSuite.addTest(unittest.makeSuite(iterTest))
    if ca.numexpr_here:
        theSuite.addTest(unittest.makeSuite(evalTest))
        theSuite.addTest(unittest.makeSuite(eval_getitemTest))
        theSuite.addTest(unittest.makeSuite(bool_getitemTest))
        theSuite.addTest(unittest.makeSuite(getif_smallTest))
        theSuite.addTest(unittest.makeSuite(getif_largeTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
