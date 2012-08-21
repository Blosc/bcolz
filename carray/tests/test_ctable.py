########################################################################
#
#       License: BSD
#       Created: September 1, 2010
#       Author:  Francesc Alted - francesc@continuum.com
#
########################################################################

import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
import unittest
from carray.tests import common
from common import MayBeDiskTest


class createTest(MayBeDiskTest):

    def test00a(self):
        """Testing ctable creation from a tuple of carrays"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing ctable creation from a tuple of lists"""
        t = ca.ctable(([1,2,3],[4,5,6]), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([[1,2,3],[4,5,6]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable creation from a tuple of carrays (single column)"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        self.assertRaises(ValueError, ca.ctable, a, 'f0', rootdir=self.rootdir)

    def test01(self):
        """Testing ctable creation from a tuple of numpy arrays"""
        N = 1e1
        a = np.arange(N, dtype='i4')
        b = np.arange(N, dtype='f8')+1
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([a,b]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing ctable creation from an structured array"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing ctable creation from large iterator"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8',
                        count=N, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing ctable creation from large iterator (with a hint)"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)),
                         dtype='i4,f8', count=N)
        t = ca.fromiter(((i, i*2.) for i in xrange(N)),
                        dtype='i4,f8', count=N, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

class createDiskTest(createTest):
    disk = True


class persistentTest(MayBeDiskTest):

    disk = True

    def test00a(self):
        """Testing ctable opening in "r" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='r')
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        self.assertRaises(RuntimeError, t.__setitem__, 1, (0, 0.0))
        self.assertRaises(RuntimeError, t.append, (0, 0.0))

    def test00b(self):
        """Testing ctable opening in "w" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='w')
        #print "t->", `t`
        N = 0
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((0, 0.0))
        t.append((0, 0.0))
        t[1] = (1, 2.0)
        ra = np.rec.fromarrays([(0,1),(0.0, 2.0)], 'i4,f8').view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable opening in "a" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='a')
        #print "t->", `t`

        # Check values
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[-1] = (11, 12.0)

        # Check values
        N = 12
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing ctable creation in "r" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        self.assertRaises(RuntimeError, ca.ctable, (a, b), ('f0', 'f1'),
                          rootdir=self.rootdir, mode='r')

    def test01b(self):
        """Testing ctable creation in "w" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Overwrite the last ctable
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir, mode='w')
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[11] = (11, 12.0)

        # Check values
        N = 12
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01c(self):
        """Testing ctable creation in "a" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Overwrite the last ctable
        self.assertRaises(RuntimeError, ca.ctable, (a, b), ('f0', 'f1'),
                          rootdir=self.rootdir, mode='a')


class add_del_colTest(MayBeDiskTest):

    def test00a(self):
        """Testing adding a new column (list flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00(self):
        """Testing adding a new column (carray flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, cparams=ca.cparams(1), rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        self.assert_(t['f2'].cparams.clevel == 1, "Incorrect clevel")

    def test02(self):
        """Testing adding a new column (default naming)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.delcol('f1')
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

class add_del_colDiskTest(add_del_colTest):
    disk = True


class getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with only a start"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start = 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start], ra[start], "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with start, stop"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start, stop = 3, 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start:stop], ra[start:stop],
                           "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with start, stop, step"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start, stop, step = 3, 9, 2
        #print "t->", `t[start:stop:step]`
        #print "ra->", ra[start:stop:step]
        assert_array_equal(t[start:stop:step], ra[start:stop:step],
                           "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with a column name"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        colname = "f1"
        #print "t->", `t[colname]`
        #print "ra->", ra[colname]
        assert_array_equal(t[colname][:], ra[colname],
                           "ctable values are not correct")

    def test04(self):
        """Testing __getitem__ with a list of column names"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        colnames = ["f0", "f2"]
        # For some version of NumPy (> 1.7) I cannot make use of
        # ra[colnames]   :-/
        ra2 = np.fromiter(((i, i*3) for i in xrange(N)), dtype='i4,i8')
        ra2.dtype.names = ('f0', 'f2')
        #print "t->", `t[colnames]`
        #print "ra2->", ra2
        assert_array_equal(t[colnames][:], ra2,
                           "ctable values are not correct")

class getitemDiskTest(getitemTest):
    disk = True


class setitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __setitem__ with only a start"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
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
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
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
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
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
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
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
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1,43, 20)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

class setitemDiskTest(setitemTest):
    disk = True


class appendTest(MayBeDiskTest):

    def test00(self):
        """Testing append() with scalar values"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.append((N, N*2))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+1)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing append() with numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((a, b))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing append() with carrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((ca.carray(a), ca.carray(b)))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing append() with structured arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t.append(ra2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing append() with another ctable"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t2 = ca.ctable(ra2)
        t.append(t2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

class appendDiskTest(appendTest):
    disk = True


class trimTest(MayBeDiskTest):

    def test00(self):
        """Testing trim() with Python scalar values"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N-2)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                       rootdir=self.rootdir)
        t.trim(2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing trim() with NumPy scalar values"""
        N = 10000
        ra = np.fromiter(((i, i*2.) for i in xrange(N-200)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.trim(np.int(200))
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing trim() with a complete trim"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(0)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.trim(N)
        self.assert_(len(ra) == len(t), "Lengths are not equal")

class trimDiskTest(trimTest):
    disk = True


class resizeTest(MayBeDiskTest):

    def test00(self):
        """Testing resize() (decreasing)"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N-2)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.resize(N-2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing resize() (increasing)"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N+4)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.resize(N+4)
        ra['f0'][N:] = np.zeros(4)
        ra['f1'][N:] = np.zeros(4)
        assert_array_equal(t[:], ra, "ctable values are not correct")

class resizeDiskTest(resizeTest):
    disk=True


class copyTest(MayBeDiskTest):

    def test00(self):
        """Testing copy() without params"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            rootdir = self.rootdir + "-test00"
        else:
            rootdir = self.rootdir
        t2 = t.copy(rootdir=rootdir, mode='w')
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            # Copy over the same location should give an error
            self.assertRaises(RuntimeError,
                              t.copy,cparams=ca.cparams(clevel=9),
                              rootdir=self.rootdir, mode='w')
            return
        else:
            t2 = t.copy(cparams=ca.cparams(clevel=9),
                        rootdir=self.rootdir, mode='w')
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t.cparams.clevel == ca.cparams().clevel)
        self.assert_(t2.cparams.clevel == 9)
        self.assert_(t['f1'].cbytes > t2['f1'].cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lower clevel"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        # print "t:", t, t.rootdir
        t2 = t.copy(cparams=ca.cparams(shuffle=False), rootdir=self.rootdir)
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")

class copyDiskTest(copyTest):
    disk = True


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


class evalTest(MayBeDiskTest):

    vm = "python"

    def setUp(self):
        self.prev_vm = ca.defaults.eval_vm
        ca.defaults.eval_vm = self.vm
        MayBeDiskTest.setUp(self)

    def tearDown(self):
        ca.defaults.eval_vm = self.prev_vm
        MayBeDiskTest.tearDown(self)

    def test00a(self):
        """Testing eval() with only columns"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 * f1 * f2")
        rar = ra['f0'] * ra['f1'] * ra['f2']
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test00b(self):
        """Testing eval() with only constants"""
        f0, f1, f2 = 1, 2, 3
        # Populate the name space with functions from math
        from math import sin
        ctr = ca.eval("f0 * f1 * sin(f2)")
        rar = f0 * f1 * sin(f2)
        #print "ctable ->", ctr
        #print "python ->", rar
        self.assert_(ctr == rar, "values are not correct")

    def test01(self):
        """Testing eval() with columns and constants"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 * f1 * 3")
        rar = ra['f0'] * ra['f1'] * 3
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test02(self):
        """Testing eval() with columns, constants and other variables"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        if not ca.defaults.eval_vm == "numexpr":
            # Populate the name space with functions from numpy
            from numpy import sin
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 >= f1")
        rar = ra['f0'] >= ra['f1']
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test05(self):
        """Testing eval() with a mix of columns and numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N)
        b = np.arange(N)
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test06(self):
        """Testing eval() with a mix of columns, numpy arrays and carrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N)
        b = ca.arange(N)
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        #print "ctable ->", ctr
        #print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

class evalDiskTest(evalTest):
    disk = True

class eval_ne(evalTest):
    vm = "numexpr"

class eval_neDisk(evalTest):
    vm = "numexpr"
    disk = True


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
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02b(self):
        """Testing fancy indexing (setitem) with a boolean array (II)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(10, size=1000).astype('bool')
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing fancy indexing (setitem) with a boolean array (all false)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.zeros(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing fancy indexing (setitem) with a boolean array (all true)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.ones(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04a(self):
        """Testing fancy indexing (setitem) with a condition (all false)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = "f0<0"
        sl2 = ra['f0'] < 0
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04b(self):
        """Testing fancy indexing (setitem) with a condition (all true)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = "f0>=0"
        sl2 = ra['f0'] >= 0
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04c(self):
        """Testing fancy indexing (setitem) with a condition (mixed values)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = "(f0>0) & (f1 < 10)"
        sl2 = (ra['f0'] > 0) & (ra['f1'] < 10)
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04d(self):
        """Testing fancy indexing (setitem) with a condition (diff values)"""
        N = 100
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = "(f0>0) & (f1 < 10)"
        sl2 = (ra['f0'] > 0) & (ra['f1'] < 10)
        l = len(np.where(sl2)[0])
        t[sl] = [(-i, -i*2., -i*3) for i in xrange(l)]
        ra[sl2] = [(-i, -i*2., -i*3) for i in xrange(l)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")


class iterTest(MayBeDiskTest):

    def test00(self):
        """Testing ctable.__iter__"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test01(self):
        """Testing ctable.iter() without params"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter()]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test02(self):
        """Testing ctable.iter() with start,stop,step"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,3)]
        nl = [r['f1'] for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test03(self):
        """Testing ctable.iter() with outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [tuple(r) for r in t.iter(outcols='f2, nrow__, f0')]
        nl = [(r['f2'], i, r['f0']) for i, r in enumerate(ra)]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test04(self):
        """Testing ctable.iter() with start,stop,step and outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r for r in t.iter(1,9,3, 'f2, nrow__ f0')]
        nl = [(r['f2'], r['f0'], r['f0']) for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test05(self):
        """Testing ctable.iter() with start, stop, step and limit"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, limit=3)]
        nl = [r['f1'] for r in ra[1:9:2][:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test06(self):
        """Testing ctable.iter() with start, stop, step and skip"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, skip=3)]
        nl = [r['f1'] for r in ra[1:9:2][3:]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test07(self):
        """Testing ctable.iter() with start, stop, step and limit, skip"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, limit=2, skip=1)]
        nl = [r['f1'] for r in ra[1:9:2][1:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

class iterDiskTest(iterTest):
    disk = True


class eval_getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with an expression (all false values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        # In t['f1*4 >= ppp'], 'ppp' variable name should be found
        self.assertRaises(NameError, t.__getitem__, 'f1*4 >= ppp')

    def test04a(self):
        """Testing __getitem__ with an expression with columns and ndarrays"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        f1 = t['f2']
        f2 = t['f1']
        rt = t['f2*4 >= f1*2']
        rar = np.fromiter(((i, i*2., i*3) for i in xrange(N) if i*4 >= i*2.*2),
                          dtype='i4,f8,i8')
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

class eval_getitemDiskTest(eval_getitemTest):
    disk = True


class bool_getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with a boolean array (all false values)"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
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
        t = ca.ctable(ra, rootdir=self.rootdir)
        barr = np.zeros(len(t)-1, dtype=np.bool_)
        self.assertRaises(IndexError, t.__getitem__, barr)

class bool_getitemDiskTest(bool_getitemTest):
    disk = True


class whereTest(MayBeDiskTest):

    def test00a(self):
        """Testing where() with a boolean array (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1 > f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test00b(self):
        """Testing where() with a boolean array (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1 <= f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if i <= i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test00c(self):
        """Testing where() with a boolean array (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('4+f1 > f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test01a(self):
        """Testing where() with an expression (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('f1 > f2')]
        rl = [i for i in xrange(N) if i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test01b(self):
        """Testing where() with an expression (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('f1 <= f2')]
        rl = [i for i in xrange(N) if i <= i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test01c(self):
        """Testing where() with an expression (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('4+f1 > f2')]
        rl = [i for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test02a(self):
        """Testing where() with an expression (with outcols)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r.f1 for r in t.where('4+f1 > f2', outcols='f1')]
        rl = [i*2. for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test02b(self):
        """Testing where() with an expression (with outcols II)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [(r.f1, r.f2) for r in t.where('4+f1 > f2', outcols=['f1','f2'])]
        rl = [(i*2., i*3) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test02c(self):
        """Testing where() with an expression (with outcols III)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [(f2, f0) for f0,f2 in t.where('4+f1 > f2', outcols='f0,f2')]
        rl = [(i*3, i) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    # This does not work anymore because of the nesting of ctable._iter
    def _test02d(self):
        """Testing where() with an expression (with outcols IV)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        where = t.where('f1 > f2', outcols='f3,  f0')
        self.assertRaises(ValueError, where.next)

    def test03(self):
        """Testing where() with an expression (with nrow__ in outcols)"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__','f2','f0'])]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt, type(rt[0][0])
        #print "rl->", rl, type(rl[0][0])
        self.assert_(rt == rl, "where not working correctly")

    def test04(self):
        """Testing where() after an iter()"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        tmp = [r for r in t.iter(1,10,3)]
        rt = [tuple(r) for r in t.where('4+f1 > f2',
                                        outcols=['nrow__','f2','f0'])]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2]
        #print "rt->", rt, type(rt[0][0])
        #print "rl->", rl, type(rl[0][0])
        self.assert_(rt == rl, "where not working correctly")

    def test05(self):
        """Testing where() with limit"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__','f2','f0'],
                                 limit=3)]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2][:3]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test06(self):
        """Testing where() with skip"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__','f2','f0'],
                                 skip=3)]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2][3:]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

    def test07(self):
        """Testing where() with limit & skip"""
        N = self.N
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__','f2','f0'],
                                 limit=1, skip=2)]
        rl = [(i, i*3, i) for i in xrange(N) if 4+i > i*2][2:3]
        #print "rt->", rt
        #print "rl->", rl
        self.assert_(rt == rl, "where not working correctly")

class where_smallTest(whereTest):
    N = 10

class where_largeTest(whereTest):
    N = 10*1000

class where_smallDiskTest(whereTest):
    N = 10
    disk = True

class where_largeDiskTest(whereTest):
    N = 10*1000
    disk = True


# This test goes here until a new test_toplevel.py would be created
class walkTest(MayBeDiskTest):
    disk = True
    ncas = 3  # the number of carrays per level
    ncts = 4  # the number of ctables per level
    nlevels = 5 # the number of levels

    def setUp(self):
        import os, os.path
        N = 10

        MayBeDiskTest.setUp(self)
        base = self.rootdir
        os.mkdir(base)

        # Create a small object hierarchy on-disk
        for nlevel in range(self.nlevels):
            newdir = os.path.join(base, 'level%s' % nlevel) 
            os.mkdir(newdir)
            for nca in range(self.ncas):
                newca = os.path.join(newdir, 'ca%s' % nca) 
                ca.zeros(N, rootdir=newca)
            for nct in range(self.ncts):
                newca = os.path.join(newdir, 'ct%s' % nct) 
                ca.fromiter(((i, i*2) for i in range(N)), count=N,
                            dtype='i2,f4',
                            rootdir=newca)
            base = newdir

    def test00(self):
        """Checking the walk toplevel function (no classname)"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in ca.walk(self.rootdir):
            if type(node) == ca.carray:
                ncas_ += 1
            elif type(node) == ca.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assert_(ncas_ == self.ncas * self.nlevels)
        self.assert_(ncts_ == self.ncts * self.nlevels)
        self.assert_(others == 0)

    def test01(self):
        """Checking the walk toplevel function (classname='carray')"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in ca.walk(self.rootdir, classname='carray'):
            if type(node) == ca.carray:
                ncas_ += 1
            elif type(node) == ca.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assert_(ncas_ == self.ncas * self.nlevels)
        self.assert_(ncts_ == 0)
        self.assert_(others == 0)

    def test02(self):
        """Checking the walk toplevel function (classname='ctable')"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in ca.walk(self.rootdir, classname='ctable'):
            if type(node) == ca.carray:
                ncas_ += 1
            elif type(node) == ca.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assert_(ncas_ == 0)
        self.assert_(ncts_ == self.ncts * self.nlevels)
        self.assert_(others == 0)



def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(createTest))
    theSuite.addTest(unittest.makeSuite(createDiskTest))
    theSuite.addTest(unittest.makeSuite(persistentTest))
    theSuite.addTest(unittest.makeSuite(add_del_colTest))
    theSuite.addTest(unittest.makeSuite(add_del_colDiskTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))
    theSuite.addTest(unittest.makeSuite(getitemDiskTest))
    theSuite.addTest(unittest.makeSuite(setitemTest))
    theSuite.addTest(unittest.makeSuite(setitemDiskTest))
    theSuite.addTest(unittest.makeSuite(appendTest))
    theSuite.addTest(unittest.makeSuite(appendDiskTest))
    theSuite.addTest(unittest.makeSuite(trimTest))
    theSuite.addTest(unittest.makeSuite(trimDiskTest))
    theSuite.addTest(unittest.makeSuite(resizeTest))
    theSuite.addTest(unittest.makeSuite(resizeDiskTest))
    theSuite.addTest(unittest.makeSuite(copyTest))
    theSuite.addTest(unittest.makeSuite(copyDiskTest))
    theSuite.addTest(unittest.makeSuite(specialTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_getitemTest))
    theSuite.addTest(unittest.makeSuite(fancy_indexing_setitemTest))
    theSuite.addTest(unittest.makeSuite(iterTest))
    theSuite.addTest(unittest.makeSuite(iterDiskTest))
    theSuite.addTest(unittest.makeSuite(evalTest))
    theSuite.addTest(unittest.makeSuite(evalDiskTest))
    if ca.numexpr_here:
        theSuite.addTest(unittest.makeSuite(eval_ne))
        theSuite.addTest(unittest.makeSuite(eval_neDisk))
    theSuite.addTest(unittest.makeSuite(eval_getitemTest))
    theSuite.addTest(unittest.makeSuite(eval_getitemDiskTest))
    theSuite.addTest(unittest.makeSuite(bool_getitemTest))
    theSuite.addTest(unittest.makeSuite(bool_getitemDiskTest))
    theSuite.addTest(unittest.makeSuite(where_smallTest))
    theSuite.addTest(unittest.makeSuite(where_smallDiskTest))
    theSuite.addTest(unittest.makeSuite(where_largeTest))
    theSuite.addTest(unittest.makeSuite(where_largeDiskTest))
    theSuite.addTest(unittest.makeSuite(walkTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
