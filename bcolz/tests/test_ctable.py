########################################################################
#
#       License: BSD
#       Created: September 1, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

import sys
import os
import tempfile

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from bcolz.tests.common import (
        MayBeDiskTest, TestCase, unittest, skipUnless, SkipTest ) 
import bcolz
from bcolz.py2help import xrange, PY2
from bcolz.py2help_tests import Mock
import pickle


class createTest(MayBeDiskTest):

    def test00a(self):
        """Testing ctable creation from a tuple of carrays"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # print "t->", `t`
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing ctable creation from a tuple of lists"""
        t = bcolz.ctable(([1, 2, 3], [4, 5, 6]), ('f0', 'f1'),
                         rootdir=self.rootdir)
        # print "t->", `t`
        ra = np.rec.fromarrays([[1, 2, 3], [4, 5, 6]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable creation from a tuple of carrays (single column)"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        self.assertRaises(ValueError, bcolz.ctable, a, 'f0',
                          rootdir=self.rootdir)

    def test01(self):
        """Testing ctable creation from a tuple of numpy arrays"""
        N = 1e1
        a = np.arange(N, dtype='i4')
        b = np.arange(N, dtype='f8') + 1
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # print "t->", `t`
        ra = np.rec.fromarrays([a, b]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing ctable creation from an structured array"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing ctable creation from large iterator"""
        N = 10 * 1000
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.fromiter(
            ((i, i * 2.) for i in xrange(N)), dtype='i4,f8', count=N,
            rootdir=self.rootdir)
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing ctable creation from large iterator (with a hint)"""
        N = 10 * 1000
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)),
                         dtype='i4,f8', count=N)
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)),
                           dtype='i4,f8', count=N, rootdir=self.rootdir)
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing freeing memory after reading (just check the API)"""
        N = 10 * 1000
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)),
                         dtype='i4,f8', count=N)
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)),
                           dtype='i4,f8', count=N, rootdir=self.rootdir)
        mt = t[:]
        t.free_cachemem()

    def test05(self):
        """Testing unicode string in column names. """
        t = bcolz.ctable(([1], [2]), (u'f0', u'f1'), rootdir=self.rootdir)
        # this should not raise an error
        t[u'f0'].rootdir

    def test06a(self):
        """Test create empty ctable"""
        N = 0
        dtype = "i4,i8,f8"
        ra = np.zeros(N, dtype=dtype)
        ct = bcolz.zeros(N, dtype=dtype, rootdir=self.rootdir)
        assert_array_equal(ct[:], ra, "ctable values are not correct")

    def test06b(self):
        """Test create empty ctable and assing names to their columns"""
        N = 0
        dtype = np.dtype(
            [('Alice', np.int16), ('Bob', np.int8), ('Charlie', np.float)])
        ra = np.zeros(N, dtype=dtype)
        ct = bcolz.zeros(N, dtype=dtype, rootdir=self.rootdir)
        self.assertEquals(ct.names, ['Alice', 'Bob', 'Charlie'])
        assert_array_equal(ct[:], ra, "ctable values are not correct")

    def test06c(self):
        """Test create empty ctable and set some cparams"""
        N = 0
        dtype = "i4,i8,f8"
        ra = np.zeros(N, dtype=dtype)
        cparams = bcolz.cparams(clevel=9, shuffle=False)
        ct = bcolz.zeros(N, dtype=dtype, cparams=cparams, rootdir=self.rootdir)
        assert_array_equal(ct[:], ra, "ctable values are not correct")
        self.assertEqual(cparams, ct.cparams)

    def test06d(self):
        """Test create empty ctable and set expectedlen"""
        N = 0
        expectedlen = int(1e7)
        ct = bcolz.zeros(0, dtype="i4,i8,f8", expectedlen=expectedlen)
        self.assertEqual(131072, ct['f0'].chunklen)
        self.assertEqual(65536, ct['f1'].chunklen)
        self.assertEqual(65536, ct['f2'].chunklen)

    def test07a(self):
        """Test create ctable full of zeros"""
        N = 10000
        dtype = "i4,i8,f4"
        ra = np.zeros(N, dtype=dtype)
        ct = bcolz.zeros(N, dtype=dtype, rootdir=self.rootdir)
        assert_array_equal(ct[:], ra, "ctable values are not correct")

    def test07b(self):
        """Test create ctable full of zeros and assign names to their columns"""
        N = 10000
        dtype = np.dtype(
            [('Alice', np.int16), ('Bob', np.int8), ('Charlie', np.float)])
        ra = np.zeros(N, dtype=dtype)
        ct = bcolz.zeros(N, dtype=dtype, rootdir=self.rootdir)
        self.assertEquals(ct.names, ['Alice', 'Bob', 'Charlie'])
        assert_array_equal(ct[:], ra, "ctable values are not correct")

    def test07c(self):
        """Test create ctable full of zeros and set some cparams"""
        N = 10000
        dtype = "i4,i8,f8"
        ra = np.zeros(N, dtype=dtype)
        cparams = bcolz.cparams(clevel=9, shuffle=False)
        ct = bcolz.zeros(N, dtype=dtype, cparams=cparams, rootdir=self.rootdir)
        assert_array_equal(ct[:], ra, "ctable values are not correct")
        self.assertEqual(cparams, ct.cparams)



class createMemoryTest(createTest, TestCase):
    disk = False


class createDiskTest(createTest, TestCase):
    disk = True


class persistentTest(MayBeDiskTest, TestCase):

    disk = True

    def test00a(self):
        """Testing ctable opening in "r" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = bcolz.open(rootdir=self.rootdir, mode='r')
        # print "t->", `t`
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        self.assertRaises(IOError, t.__setitem__, 1, (0, 0.0))
        self.assertRaises(IOError, t.append, (0, 0.0))

    def test00b(self):
        """Testing ctable opening in "w" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Opening in 'w'rite mode is not allowed.  First remove the file.
        self.assertRaises(ValueError, bcolz.open,
                          rootdir=self.rootdir, mode='w')

    def test00c(self):
        """Testing ctable opening in "a" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = bcolz.open(rootdir=self.rootdir, mode='a')
        # print "t->", `t`

        # Check values
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[-1] = (11, 12.0)

        # Check values
        N = 12
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing ctable creation in "r" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir, mode='w')
        t2 = bcolz.ctable(rootdir=self.rootdir, mode='r')
        self.assertRaises(IOError, t2.append, (a, b))

    def test01b(self):
        """Testing ctable creation in "w" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Overwrite the last ctable
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir, mode='w')
        # print "t->", `t`
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[11] = (11, 12.0)

        # Check values
        N = 12
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01c(self):
        """Testing ctable re-opening in "a" mode"""
        N = 1e1
        a = bcolz.carray(np.arange(N, dtype='i4'))
        b = bcolz.carray(np.arange(N, dtype='f8') + 1)
        t = bcolz.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Append to the last ctable
        bcolz.ctable(rootdir=self.rootdir, mode='a')
        ra = np.rec.fromarrays([a[:], b[:]]).view(np.ndarray)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01d(self):
        """Testing ctable opening in "r" mode with nonexistent directory"""
        tempdir = tempfile.mkdtemp(prefix='bcolz-test01d')
        non_existent_root = os.path.join(tempdir, 'not/a/real/path')
        expected_message = (
            "Disk-based ctable opened with `r`ead mode "
            "yet `rootdir='{rootdir}'` does not exist".format(
                rootdir=non_existent_root,
            )
        )

        with self.assertRaises(KeyError) as ctx:
            bcolz.ctable(rootdir=non_existent_root, mode='r')
        self.assertEqual(ctx.exception.args[0], expected_message)


class add_del_colTest(MayBeDiskTest):

    def test00a(self):
        """Testing adding a new column (list flavor)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.fromiter(("s%d" % i for i in xrange(N)), dtype='S2')
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i * 2., "s%d" % i) for i in xrange(N)),
                         dtype='i4,f8,S2')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00(self):
        """Testing adding a new column (carray flavor)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(bcolz.carray(c), 'f2')
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(c, 'f2')
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01b(self):
        """Testing cparams when adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, cparams=bcolz.cparams(1), rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(c, 'f2')
        self.assertTrue(t['f2'].cparams.clevel == 1, "Incorrect clevel")

    def test02(self):
        """Testing adding a new column (default naming)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(bcolz.carray(c))
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing inserting a new column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(c, name='c0', pos=0)
        ra = np.fromiter(((i * 3, i, i * 2.)
                          for i in xrange(N)), dtype='i8,i4,f8')
        ra.dtype.names = ('c0', 'f0', 'f1')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing inserting a new column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8') * 3
        t.addcol(c, name='c0', pos=1)
        ra = np.fromiter(((i, i * 3, i * 2.)
                          for i in xrange(N)), dtype='i4,i8,f8')
        ra.dtype.names = ('f0', 'c0', 'f1')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test05(self):
        """Testing removing an existing column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i * 3, i * 2.)
                          for i in xrange(N)), dtype='i4,i8,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=0)
        # The next gives a segfault.  See:
        # http://projects.scipy.org/numpy/ticket/1598
        # ra = np.fromiter(((i*3, i*2) for i in xrange(N)), dtype='i8,f8')
        # ra.dtype.names = ('f1', 'f2')
        dt = np.dtype([('f1', 'i8'), ('f2', 'f8')])
        ra = np.fromiter(((i * 3, i * 2) for i in xrange(N)), dtype=dt)
        # print "t->", `t`
        # print "ra", ra
        # assert_array_equal(t[:], ra, "ctable values are not correct")

    def test06(self):
        """Testing removing an existing column (at the end)"""
        N = 10
        ra = np.fromiter(((i, i * 3, i * 2.)
                          for i in xrange(N)), dtype='i4,i8,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=2)
        ra = np.fromiter(((i, i * 3) for i in xrange(N)), dtype='i4,i8')
        ra.dtype.names = ('f0', 'f1')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test07(self):
        """Testing removing an existing column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i * 3, i * 2.)
                          for i in xrange(N)), dtype='i4,i8,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=1)
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test08(self):
        """Testing removing an existing column (by name)"""
        N = 10
        ra = np.fromiter(((i, i * 3, i * 2.)
                          for i in xrange(N)), dtype='i4,i8,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.delcol('f1')
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")


class add_del_colMemoryTest(add_del_colTest, TestCase):
    disk = False


class add_del_colDiskTest(add_del_colTest, TestCase):
    disk = True

    def test_add_new_column_ondisk(self):
        """Testing adding a new column properly creates a new disk array (list
        flavor)
        """
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.fromiter(("s%d" % i for i in xrange(N)), dtype='S2')
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i * 2., "s%d" % i) for i in xrange(N)),
                         dtype='i4,f8,S2')
        newpath = os.path.join(self.rootdir, 'f2')
        assert_array_equal(t[:], ra, "ctable values are not correct")
        assert_array_equal(bcolz.carray(rootdir=newpath)[:], ra['f2'])

    def test_del_new_column_ondisk(self):
        """Testing delcol removes data on disk.
        """
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.fromiter(("s%d" % i for i in xrange(N)), dtype='S2')
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i * 2., "s%d" % i) for i in xrange(N)),
                         dtype='i4,f8,S2')
        newpath = os.path.join(self.rootdir, 'f2')
        assert_array_equal(t[:], ra, "ctable values are not correct")
        assert_array_equal(bcolz.carray(rootdir=newpath)[:], ra['f2'])
        t.delcol('f2')
        self.assertFalse(os.path.exists(newpath))

    def test_del_new_column_ondisk(self):
        """Testing delcol with keep keeps the data.
        """
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.fromiter(("s%d" % i for i in xrange(N)), dtype='S2')
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i * 2., "s%d" % i) for i in xrange(N)),
                         dtype='i4,f8,S2')
        newpath = os.path.join(self.rootdir, 'f2')
        assert_array_equal(t[:], ra, "ctable values are not correct")
        assert_array_equal(bcolz.carray(rootdir=newpath)[:], ra['f2'])
        t.delcol('f2', keep=True)
        self.assertTrue(os.path.exists(newpath))

    def test_add_new_column_ondisk_other_carray_rootdir(self):
        """Testing addcol with different rootdir.
        """
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c = np.fromiter(("s%d" % i for i in xrange(N)), dtype='S2')
        temp_dir = os.path.join(tempfile.mkdtemp('bcolz-'), 'c')
        c = bcolz.carray(c, rootdir=temp_dir)
        t.addcol(c, 'f2')
        ra = np.fromiter(((i, i * 2., "s%d" % i) for i in xrange(N)),
                         dtype='i4,f8,S2')
        newpath = os.path.join(self.rootdir, 'f2')
        assert_array_equal(t[:], ra, "ctable values are not correct")
        assert_array_equal(bcolz.carray(rootdir=newpath)[:], ra['f2'])
        self.assertEqual(temp_dir, c.rootdir)
        self.assertEqual(newpath, t['f2'].rootdir)


class getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with only a start"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        start = 9
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[start], ra[start],
                           "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with start, stop"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        start, stop = 3, 9
        # print "t->", `t`
        # print "ra[:]", ra[:]
        assert_array_equal(t[start:stop], ra[start:stop],
                           "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with start, stop, step"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        start, stop, step = 3, 9, 2
        # print "t->", `t[start:stop:step]`
        # print "ra->", ra[start:stop:step]
        assert_array_equal(t[start:stop:step], ra[start:stop:step],
                           "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with a column name"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        colname = "f1"
        # print "t->", `t[colname]`
        # print "ra->", ra[colname]
        assert_array_equal(t[colname][:], ra[colname],
                           "ctable values are not correct")

    def test04(self):
        """Testing __getitem__ with a list of column names"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        colnames = ["f0", "f2"]
        # For some version of NumPy (> 1.7) I cannot make use of
        # ra[colnames]   :-/
        ra2 = np.fromiter(((i, i * 3) for i in xrange(N)), dtype='i4,i8')
        ra2.dtype.names = ('f0', 'f2')
        # print "t->", `t[colnames]`
        # print "ra2->", ra2
        assert_array_equal(t[colnames][:], ra2,
                           "ctable values are not correct")

    def test05a(self):
        """Testing __getitem__ with total slice"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        assert_array_equal(t[:], ra[:],
                           "ctable values are not correct")

    def test05b(self):
        """Testing __getitem__ with total slice for table including a
        multidimensional column"""
        N = 10
        # N.B., col1 is 2D
        ra = np.fromiter(((i, (i * 2., i * 4.))
                          for i in xrange(N)),
                          dtype=[('col0', 'i4'), ('col1', ('f8', 2))])
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        assert_array_equal(t[:], ra[:],
                           "ctable values are not correct")


class getitemMemoryTest(getitemTest, TestCase):
    disk = False


class getitemDiskTest(getitemTest, TestCase):
    disk = True


class setitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __setitem__ with only a start"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing __setitem__ with only a stop"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(None, 9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing __setitem__ with a start, stop"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 90, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing __setitem__ with a start, stop, step"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 90, 2)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing __setitem__ with a large step"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1, 43, 20)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")


class setitemMemoryTest(setitemTest, TestCase):
    disk = False


class setitemDiskTest(setitemTest, TestCase):
    disk = True


class appendTest(MayBeDiskTest):

    def test00a(self):
        """Testing append() with scalar values"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.append((N, N * 2))
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 1)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing append() with a list of scalar values"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t.append([[N, N + 1], [N * 2, (N + 1) * 2]])
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 2)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing append() with numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N + 10, dtype='i4')
        b = np.arange(N, N + 10, dtype='f8') * 2.
        t.append((a, b))
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing append() with carrays"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N + 10, dtype='i4')
        b = np.arange(N, N + 10, dtype='f8') * 2.
        t.append((bcolz.carray(a), bcolz.carray(b)))
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing append() with structured arrays"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i * 2.)
                           for i in xrange(N, N + 10)), dtype='i4,f8')
        t.append(ra2)
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing append() with another ctable"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i * 2.)
                           for i in xrange(N, N + 10)), dtype='i4,f8')
        t2 = bcolz.ctable(ra2)
        t.append(t2)
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test05(self):
        """Testing append() with void types"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra[:-1], rootdir=self.rootdir)
        t.append(ra[-1])
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test06(self):
        """Extracting rows from table with np.object column"""
        N = 4
        dtype = np.dtype([("a", np.object), ("b", np.uint8), ("c", np.int32), 
            ("d", np.float32) ])
        with bcolz.ctable(np.empty(0, dtype=dtype), rootdir=self.rootdir) as t:
            for i in xrange(N):
                t.append((str(i), i*2, i*4, i*8))
            result = t[np.array([1, 0, 2])]
            assert_array_equal(result[0], t[1])
            assert_array_equal(result[1], t[0])
            assert_array_equal(result[2], t[2])


class appendMemoryTest(appendTest, TestCase):
    disk = False


class appendDiskTest(appendTest, TestCase):
    disk = True


class trimTest(MayBeDiskTest):

    def test00(self):
        """Testing trim() with Python scalar values"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N - 2)), dtype='i4,f8')
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)), 'i4,f8', N,
                           rootdir=self.rootdir)
        t.trim(2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing trim() with NumPy scalar values"""
        N = 10000
        ra = np.fromiter(((i, i * 2.) for i in xrange(N - 200)), dtype='i4,f8')
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)), 'i4,f8', N,
                           rootdir=self.rootdir)
        t.trim(np.int(200))
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing trim() with a complete trim"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(0)), dtype='i4,f8')
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)), 'i4,f8', N,
                           rootdir=self.rootdir)
        t.trim(N)
        self.assertTrue(len(ra) == len(t), "Lengths are not equal")


class trimMemoryTest(trimTest, TestCase):
    disk = False


class trimDiskTest(trimTest, TestCase):
    disk = True


class resizeTest(MayBeDiskTest):

    def test00(self):
        """Testing resize() (decreasing)"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N - 2)), dtype='i4,f8')
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)), 'i4,f8', N,
                           rootdir=self.rootdir)
        t.resize(N - 2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing resize() (increasing)"""
        N = 100
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 4)), dtype='i4,f8')
        t = bcolz.fromiter(((i, i * 2.) for i in xrange(N)), 'i4,f8', N,
                           rootdir=self.rootdir)
        t.resize(N + 4)
        ra['f0'][N:] = np.zeros(4)
        ra['f1'][N:] = np.zeros(4)
        assert_array_equal(t[:], ra, "ctable values are not correct")


class resizeMemoryTest(resizeTest, TestCase):
    disk = False


class resizeDiskTest(resizeTest, TestCase):
    disk = True


class copyTest(MayBeDiskTest):

    N = 100 * 1000

    def test00(self):
        """Testing copy() without params"""
        N = 10
        ra = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            rootdir = self.rootdir + "-test00"
        else:
            rootdir = self.rootdir
        t2 = t.copy(rootdir=rootdir, mode='w')
        a = np.arange(N, N + 10, dtype='i4')
        b = np.arange(N, N + 10, dtype='f8') * 2.
        t2.append((a, b))
        ra = np.fromiter(((i, i * 2.) for i in xrange(N + 10)), dtype='i4,f8')
        self.assertTrue(len(t) == N, "copy() does not work correctly")
        self.assertTrue(len(t2) == N + 10, "copy() does not work correctly")
        assert_array_equal(t2[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing copy() with higher clevel"""
        N = self.N
        ra = np.fromiter(((i, i ** 2.2) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            # Copy over the same location should give an error
            self.assertRaises(IOError,
                              t.copy, cparams=bcolz.cparams(clevel=9),
                              rootdir=self.rootdir, mode='w')
            return
        else:
            t2 = t.copy(cparams=bcolz.cparams(clevel=9),
                        rootdir=self.rootdir, mode='w')
        # print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assertTrue(t.cparams.clevel == bcolz.cparams().clevel)
        self.assertTrue(t2.cparams.clevel == 9)
        self.assertTrue(t['f1'].cbytes > t2['f1'].cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lower clevel"""
        N = self.N
        ra = np.fromiter(((i, i ** 2.2) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        t2 = t.copy(cparams=bcolz.cparams(clevel=1))
        self.assertTrue(t.cparams.clevel == bcolz.cparams().clevel)
        self.assertTrue(t2.cparams.clevel == 1)
        # print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assertTrue(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")

    def test03(self):
        """Testing copy() with no shuffle"""
        N = self.N
        ra = np.fromiter(((i, i ** 2.2) for i in xrange(N)), dtype='i4,f8')
        t = bcolz.ctable(ra)
        # print "t:", repr(t), t.rootdir
        t2 = t.copy(cparams=bcolz.cparams(shuffle=False), rootdir=self.rootdir)
        # print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assertTrue(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")


class copyMemoryTest(copyTest, TestCase):
    disk = False


class copyDiskTest(copyTest, TestCase):
    disk = True


class specialTest(TestCase):

    def test00(self):
        """Testing __len__()"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        self.assertTrue(len(t) == len(ra),
                        "Objects do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big ctables)"""
        N = int(1e5)
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        # print "size t uncompressed ->", t.nbytes
        # print "size t compressed   ->", t.cbytes
        self.assertTrue(sys.getsizeof(t) < t.nbytes,
                        "ctable does not seem to compress at all")

    def test02(self):
        """Testing __sizeof__() (small ctables)"""
        N = int(111)
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        # print "size t uncompressed ->", t.nbytes
        # print "size t compressed   ->", t.cbytes
        self.assertTrue(sys.getsizeof(t) > t.nbytes,
                        "ctable compress too much??")


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

    def test00a(self):
        """Testing eval() with only columns"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 * f1 * f2")
        rar = ra['f0'] * ra['f1'] * ra['f2']
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test00b(self):
        """Testing eval() with only constants"""
        f0, f1, f2 = 1, 2, 3
        # Populate the name space with functions from math
        from math import sin
        ctr = bcolz.eval("f0 * f1 * sin(f2)")
        rar = f0 * f1 * sin(f2)
        # print "ctable ->", ctr
        # print "python ->", rar
        self.assertTrue(ctr == rar, "values are not correct")

    def test01(self):
        """Testing eval() with columns and constants"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 * f1 * 3")
        rar = ra['f0'] * ra['f1'] * 3
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test02(self):
        """Testing eval() with columns, constants and other variables"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        var_ = 10.
        ctr = t.eval("f0 * f2 * var_")
        rar = ra['f0'] * ra['f2'] * var_
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test03(self):
        """Testing eval() with columns and numexpr functions"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        if not bcolz.defaults.eval_vm == "numexpr":
            # Populate the name space with functions from numpy
            from numpy import sin  # noqa
        ctr = t.eval("f0 * sin(f1)")
        rar = ra['f0'] * np.sin(ra['f1'])
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_allclose(ctr[:], rar, rtol=1e-15,
                        err_msg="ctable values are not correct")

    def test04(self):
        """Testing eval() with a boolean as output"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        ctr = t.eval("f0 >= f1")
        rar = ra['f0'] >= ra['f1']
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test05(self):
        """Testing eval() with a mix of columns and numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N)
        b = np.arange(N)
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test06(self):
        """Testing eval() with a mix of columns, numpy arrays and carrays"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N)
        b = bcolz.arange(N)
        ctr = t.eval("f0 + f1 - a + b")
        rar = ra['f0'] + ra['f1'] - a + b
        # print "ctable ->", ctr
        # print "numpy  ->", rar
        assert_array_equal(ctr[:], rar, "ctable values are not correct")

    def test07(self):
        """Testing eval() with Unicode vars (via where).  Ticket #38."""
        a = np.array(['a', 'b', 'c'], dtype='U4')
        b = bcolz.ctable([a], names=['text'])
        assert [i.text for i in b.where('text == "b"')] == [u"b"]


class evalMemoryTest(evalTest, TestCase):
    disk = False


class evalDiskTest(evalTest, TestCase):
    disk = True


class eval_ne(evalTest, TestCase):
    vm = "numexpr"


class eval_neDisk(evalTest, TestCase):
    vm = "numexpr"
    disk = True


class fancy_indexing_getitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing with a small list"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        rt = t[[3, 1]]
        rar = ra[[3, 1]]
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing with a large numpy array"""
        N = 10 * 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        idx = np.random.randint(1000, size=1000)
        rt = t[idx]
        rar = ra[idx]
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing fancy indexing with an empty list"""
        N = 10 * 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        rt = t[[]]
        rar = ra[[]]
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        N = 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        rt = t[[2.3, 5.6]]
        rar = ra[[2.3, 5.6]]
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1, 101)
        b = bcolz.carray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(IndexError, b.__getitem__, idx)


class fancy_indexing_setitemTest(TestCase):

    def test00a(self):
        """Testing fancy indexing (setitem) with a small list"""
        N = 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = [3, 1]
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing fancy indexing (setitem) with a small list (II)"""
        N = 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = [3, 1]
        t[sl] = [(-1, -2, -3), (-3, -2, -1)]
        ra[sl] = [(-1, -2, -3), (-3, -2, -1)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing (setitem) with a large array"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = np.random.randint(N, size=100)
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02a(self):
        """Testing fancy indexing (setitem) with a boolean array (I)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = np.random.randint(2, size=1000).astype('bool')
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02b(self):
        """Testing fancy indexing (setitem) with a boolean array (II)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = np.random.randint(10, size=1000).astype('bool')
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing fancy indexing (setitem) with a boolean array (all false)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = np.zeros(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing fancy indexing (setitem) with a boolean array (all true)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = np.ones(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    @skipUnless(sys.version < "3", "not working in Python 3")
    def test04a(self):
        """Testing fancy indexing (setitem) with a condition (all false)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = "f0<0"
        sl2 = ra['f0'] < 0
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    @skipUnless(sys.version < "3", "not working in Python 3")
    def test04b(self):
        """Testing fancy indexing (setitem) with a condition (all true)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = "f0>=0"
        sl2 = ra['f0'] >= 0
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    @skipUnless(sys.version < "3", "not working in Python 3")
    def test04c(self):
        """Testing fancy indexing (setitem) with a condition (mixed values)"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = "(f0>0) & (f1 < 10)"
        sl2 = (ra['f0'] > 0) & (ra['f1'] < 10)
        t[sl] = [(-1, -2, -3)]
        ra[sl2] = [(-1, -2, -3)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    @skipUnless(sys.version < "3", "not working in Python 3")
    def test04d(self):
        """Testing fancy indexing (setitem) with a condition (diff values)"""
        N = 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=10)
        sl = "(f0>0) & (f1 < 10)"
        sl2 = (ra['f0'] > 0) & (ra['f1'] < 10)
        l = len(np.where(sl2)[0])
        t[sl] = [(-i, -i * 2., -i * 3) for i in xrange(l)]
        ra[sl2] = [(-i, -i * 2., -i * 3) for i in xrange(l)]
        # print "t[%s] -> %r" % (sl, t)
        # print "ra[%s] -> %r" % (sl2, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")


class iterTest(MayBeDiskTest):

    N = 10

    def test00(self):
        """Testing ctable.__iter__"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t]
        nl = [r['f1'] for r in ra]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test01(self):
        """Testing ctable.iter() without params"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter()]
        nl = [r['f1'] for r in ra]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test02(self):
        """Testing ctable.iter() with start,stop,step"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1, 9, 3)]
        nl = [r['f1'] for r in ra[1:9:3]]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test03(self):
        """Testing ctable.iter() with outcols"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [tuple(r) for r in t.iter(outcols='f2, nrow__, f0')]
        nl = [(r['f2'], i, r['f0']) for i, r in enumerate(ra)]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test04(self):
        """Testing ctable.iter() with start,stop,step and outcols"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r for r in t.iter(1, 9, 3, 'f2, nrow__ f0')]
        nl = [(r['f2'], r['f0'], r['f0']) for r in ra[1:9:3]]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test05(self):
        """Testing ctable.iter() with start, stop, step and limit"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1, 9, 2, limit=3)]
        nl = [r['f1'] for r in ra[1:9:2][:3]]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test06(self):
        """Testing ctable.iter() with start, stop, step and skip"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1, 9, 2, skip=3)]
        nl = [r['f1'] for r in ra[1:9:2][3:]]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctily")

    def test07(self):
        """Testing ctable.iter() with start, stop, step and limit, skip"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1, 9, 2, limit=2, skip=1)]
        nl = [r['f1'] for r in ra[1:9:2][1:3]]
        # print "cl ->", cl
        # print "nl ->", nl
        self.assertTrue(cl == nl, "iter not working correctly")

    def test08(self):
        """Testing several iterators in stage"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        u = t.iter(1, 9, 2, limit=2, skip=1)
        w = t.iter(1, 9, 2)
        wl = [r.f1 for r in w]
        nl = [r['f1'] for r in ra[1:9:2]]
        self.assertEqual(wl, nl, "iter not working correctly")
        ul = [r.f1 for r in u]
        nl2 = [r['f1'] for r in ra[1:9:2][1:3]]
        self.assertEqual(ul, nl2, "iter not working correctly")

    def test09(self):
        """Testing several iterators in parallel"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        u = t.iter(1, 10, 2)
        w = t.iter(1, 5, 1)
        wl = [(r[0].f1, r[1].f1) for r in zip(u, w)]
        nl = [(r[0]['f1'], r[1]['f1']) for r in zip(ra[1:10:2], ra[1:5:1])]
        self.assertEqual(wl, nl, "iter not working correctly")

    def test10a(self):
        """Testing the reuse of exhausted iterators (I)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        u = iter(t)
        wl = (r.f1 for r in u)
        nl = (r['f1'] for r in iter(ra))
        self.assertEqual(list(wl), list(nl), "iter not working correctly")
        self.assertEqual(list(wl), list(nl), "iter not working correctly")

    def test10b(self):
        """Testing the reuse of exhausted iterators (II)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, chunklen=4, rootdir=self.rootdir)
        u = t.iter(1, 10, 2)
        wl = (r.f1 for r in u)
        nl = (r['f1'] for r in iter(ra[1:10:2]))
        self.assertEqual(list(wl), list(nl), "iter not working correctly")
        self.assertEqual(list(wl), list(nl), "iter not working correctly")


class iterMemoryTest(iterTest, TestCase):
    disk = False


class iterDiskTest(iterTest, TestCase):
    disk = True


class iterblocksTest(MayBeDiskTest):

    def test00(self):
        """Testing `iterblocks` method with no blen, no start, no stop"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in bcolz.iterblocks(t):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N)
        # as per Gauss summation formula
        self.assertEqual(s, (N - 1) * (N / 2))

    def test01(self):
        """Testing `iterblocks` method with no start, no stop"""
        N, blen = self.N, 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in bcolz.iterblocks(t, blen):
            if l == 0:
                self.assertEqual(len(block), blen)
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N)
        # as per Gauss summation formula
        self.assertEqual(s, (N - 1) * (N / 2))

    def test02(self):
        """Testing `iterblocks` method with no stop"""
        N, blen = self.N, 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0.
        for block in bcolz.iterblocks(t, blen, blen - 1):
            l += len(block)
            # f8 is to small to hold the sum on 32 bit
            s += block['f1'].sum(dtype='f16')
        self.assertEqual(l, (N - (blen - 1)))
        self.assertEqual(s, (np.arange(blen - 1, N, dtype='f8') * 2).sum())

    def test03(self):
        """Testing `iterblocks` method with all parameters set"""
        N, blen = self.N, 100
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in bcolz.iterblocks(t, blen, blen - 1, 3 * blen + 2):
            l += len(block)
            s += block['f2'].sum()
        mlen = min(N - (blen - 1), 2 * blen + 3)
        self.assertEqual(l, mlen)
        slen = min(N, 3 * blen + 2)
        self.assertEqual(s, (np.arange(blen - 1, slen) * 3).sum())


class small_iterblocksMemoryTest(iterblocksTest, TestCase):
    N = 100
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


class eval_getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with an expression (all false values)"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = t['f1 > f2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i > i * 2.),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with an expression (all true values)"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = t['f1 <= f2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i <= i * 2.),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with an expression (true/false values)"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = t['f1*4 >= f2*2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i * 4 >= i * 2. * 2),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with an invalid expression"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        # In t['f1*4 >= ppp'], 'ppp' variable name should be found
        self.assertRaises(NameError, t.__getitem__, 'f1*4 >= ppp')

    def test04a(self):
        """Testing __getitem__ with an expression with columns and ndarrays"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c2 = t['f2'][:]
        rt = t['f1*4 >= c2*2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i * 4 >= i * 2. * 2),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test04b(self):
        """Testing __getitem__ with an expression with columns and carrays"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        c2 = t['f2']
        rt = t['f1*4 >= c2*2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i * 4 >= i * 2. * 2),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test05(self):
        """Testing __getitem__ with an expression with overwritten vars"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        f1 = t['f2']
        f2 = t['f1']
        rt = t['f2*4 >= f1*2']
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i * 4 >= i * 2. * 2),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")


class eval_getitemMemoryTest(eval_getitemTest, TestCase):
    disk = False


class eval_getitemDiskTest(eval_getitemTest, TestCase):
    disk = True


class bool_getitemTest(MayBeDiskTest):

    def test00(self):
        """Testing __getitem__ with a boolean array (all false values)"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1 > f2')
        rt = t[barr]
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i > i * 2.),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with a boolean array (mixed values)"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1*4 >= f2*2')
        rt = t[barr]
        rar = np.fromiter(((i, i * 2., i * 3) for i in xrange(N)
                           if i * 4 >= i * 2. * 2),
                          dtype='i4,f8,i8')
        # print "rt->", rt
        # print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with a short boolean array"""
        N = 10
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = np.zeros(len(t) - 1, dtype=np.bool_)
        self.assertRaises(IndexError, t.__getitem__, barr)


class bool_getitemMemoryTest(bool_getitemTest, TestCase):
    disk = False


class bool_getitemDiskTest(bool_getitemTest, TestCase):
    disk = True


class whereTest(MayBeDiskTest):

    def test00a(self):
        """Testing where() with a boolean array (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1 > f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test00b(self):
        """Testing where() with a boolean array (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('f1 <= f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if i <= i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test00c(self):
        """Testing where() with a boolean array (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        barr = t.eval('4+f1 > f2')
        rt = [r.f0 for r in t.where(barr)]
        rl = [i for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test01a(self):
        """Testing where() with an expression (all false values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('f1 > f2')]
        rl = [i for i in xrange(N) if i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test01b(self):
        """Testing where() with an expression (all true values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('f1 <= f2')]
        rl = [i for i in xrange(N) if i <= i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test01c(self):
        """Testing where() with an expression (mix values)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r.f0 for r in t.where('4+f1 > f2')]
        rl = [i for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test02a(self):
        """Testing where() with an expression (with outcols)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r.f1 for r in t.where('4+f1 > f2', outcols='f1')]
        rl = [i * 2. for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test02b(self):
        """Testing where() with an expression (with outcols II)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [(r.f1, r.f2) for r in t.where('4+f1 > f2', outcols=['f1', 'f2'])]
        rl = [(i * 2., i * 3) for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test02c(self):
        """Testing where() with an expression (with outcols III)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [(f2, f0) for f0, f2 in t.where('4+f1 > f2', outcols='f0,f2')]
        rl = [(i * 3, i) for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    # This does not work anymore because of the nesting of ctable._iter
    def _test02d(self):
        """Testing where() with an expression (with outcols IV)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        where = t.where('f1 > f2', outcols='f3,  f0')
        self.assertRaises(ValueError, where.next)

    def test03(self):
        """Testing where() with an expression (with nrow__ in outcols)"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__', 'f2', 'f0'])]
        rl = [(i, i * 3, i) for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt, type(rt[0][0])
        # print "rl->", rl, type(rl[0][0])
        self.assertTrue(rt == rl, "where not working correctly")

    def test04(self):
        """Testing where() after an iter()"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        tmp = [r for r in t.iter(1, 10, 3)]
        rt = [tuple(r) for r in t.where('4+f1 > f2',
                                        outcols=['nrow__', 'f2', 'f0'])]
        rl = [(i, i * 3, i) for i in xrange(N) if 4 + i > i * 2]
        # print "rt->", rt, type(rt[0][0])
        # print "rl->", rl, type(rl[0][0])
        self.assertTrue(rt == rl, "where not working correctly")

    def test05(self):
        """Testing where() with limit"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__', 'f2', 'f0'],
                                 limit=3)]
        rl = [(i, i * 3, i) for i in xrange(N) if 4 + i > i * 2][:3]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test06(self):
        """Testing where() with skip"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__', 'f2', 'f0'],
                                 skip=3)]
        rl = [(i, i * 3, i) for i in xrange(N) if 4 + i > i * 2][3:]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test07(self):
        """Testing where() with limit & skip"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra, rootdir=self.rootdir)
        rt = [r for r in t.where('4+f1 > f2', outcols=['nrow__', 'f2', 'f0'],
                                 limit=1, skip=2)]
        rl = [(i, i * 3, i) for i in xrange(N) if 4 + i > i * 2][2:3]
        # print "rt->", rt
        # print "rl->", rl
        self.assertTrue(rt == rl, "where not working correctly")

    def test08(self):
        """Testing several iterators in stage.  Ticket #37"""
        bc = bcolz.ctable([[1, 2, 3], [10, 20, 30]], names=['a', 'b'])
        u = bc.where('a >= 2')  # call .where but don't do anything with it
        self.assertEqual([10, 20, 30], list(bc['b']))

    def test09(self):
        """Testing several iterators in parallel. Ticket #37"""
        a = np.array([10, 20, 30, 40])
        bc = bcolz.ctable([[1, 2, 3, 4], [10, 20, 30, 40]], names=['a', 'b'])
        b1 = bc.where('a >= 1')
        b2 = bc.where('a >= 2')
        a1 = iter(a[a >= 10])
        a2 = iter(a[a >= 20])
        self.assertEqual([i for i in zip(a1, a2)],
                         [(i[0].b, i[1].b) for i in zip(b1, b2)])

    def test10(self):
        """Testing the reuse of exhausted iterators"""
        a = np.array([10, 20, 30, 40])
        bc = bcolz.ctable([[1, 2, 3, 4], [10, 20, 30, 40]], names=['a', 'b'])
        bi = bc.where('a >= 1')
        ai = iter(a[a >= 10])
        self.assertEqual([i for i in ai], [i.b for i in bi])
        self.assertEqual([i for i in ai], [i.b for i in bi])


class where_smallTest(whereTest, TestCase):
    N = 10


class where_largeTest(whereTest, TestCase):
    N = 10 * 1000


class where_smallDiskTest(whereTest, TestCase):
    N = 10
    disk = True


class where_largeDiskTest(whereTest, TestCase):
    N = 10 * 1000
    disk = True


# This test goes here until a new test_toplevel.py would be created
class walkTest(MayBeDiskTest, TestCase):
    disk = True
    ncas = 3  # the number of carrays per level
    ncts = 4  # the number of ctables per level
    nlevels = 5  # the number of levels

    def setUp(self):
        import os
        import os.path
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
                bcolz.zeros(N, rootdir=newca)
            for nct in range(self.ncts):
                newca = os.path.join(newdir, 'ct%s' % nct)
                bcolz.fromiter(((i, i * 2) for i in range(N)), count=N,
                               dtype='i2,f4',
                               rootdir=newca)
            base = newdir

    def test00(self):
        """Checking the walk toplevel function (no classname)"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in bcolz.walk(self.rootdir):
            if type(node) == bcolz.carray:
                ncas_ += 1
            elif type(node) == bcolz.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assertTrue(ncas_ == self.ncas * self.nlevels)
        self.assertTrue(ncts_ == self.ncts * self.nlevels)
        self.assertTrue(others == 0)

    def test01(self):
        """Checking the walk toplevel function (classname='carray')"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in bcolz.walk(self.rootdir, classname='carray'):
            if type(node) == bcolz.carray:
                ncas_ += 1
            elif type(node) == bcolz.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assertTrue(ncas_ == self.ncas * self.nlevels)
        self.assertTrue(ncts_ == 0)
        self.assertTrue(others == 0)

    def test02(self):
        """Checking the walk toplevel function (classname='ctable')"""

        ncas_, ncts_, others = (0, 0, 0)
        for node in bcolz.walk(self.rootdir, classname='ctable'):
            if type(node) == bcolz.carray:
                ncas_ += 1
            elif type(node) == bcolz.ctable:
                ncts_ += 1
            else:
                others += 1

        self.assertTrue(ncas_ == 0)
        self.assertTrue(ncts_ == self.ncts * self.nlevels)
        self.assertTrue(others == 0)


class conversionTest(TestCase):

    @skipUnless(bcolz.pandas_here, "pandas not here")
    def test00(self):
        """Testing roundtrips to a pandas dataframe"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        ct = bcolz.ctable(ra)
        df = ct.todataframe()
        ct2 = bcolz.ctable.fromdataframe(df)
        for key in ct.names:
            assert_allclose(ct2[key][:], ct[key][:])

    @skipUnless(bcolz.tables_here, "PyTables not here")
    def test01(self):
        """Testing roundtrips to a HDF5 file"""
        N = 1000
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        ct = bcolz.ctable(ra)
        tmpfile = tempfile.mktemp(".h5")
        ct.tohdf5(tmpfile)
        ct2 = bcolz.ctable.fromhdf5(tmpfile)
        self.assertEqual(ct.dtype, ct2.dtype)
        for key in ct.names:
            assert_allclose(ct2[key][:], ct[key][:])
        os.remove(tmpfile)


class pickleTest(MayBeDiskTest, TestCase):

    disk = True

    def test_pickleable(self):
        b = bcolz.ctable([[1, 2, 3], [1, 2, 3]],
                         names=['a', 'b'],
                         rootdir=self.rootdir)
        s = pickle.dumps(b)
        if PY2:
            self.assertTrue(type(s), str)
        else:
            self.assertTrue(type(s), bytes)

        b2 = pickle.loads(s)
        self.assertEquals(b2.rootdir, b.rootdir)
        self.assertEquals(type(b2), type(b))

    def test_pickleable_memory(self):
        b = bcolz.ctable([[1, 2, 3], [1, 2, 3]],
                         names=['a', 'b'],
                         rootdir=None)
        s = pickle.dumps(b)
        if PY2:
            self.assertIsInstance(s, str)
        else:
            self.assertIsInstance(s, bytes)

        b2 = pickle.loads(s)
        self.assertEquals(type(b2), type(b))


class FlushDiskTest(MayBeDiskTest, TestCase):
    disk = True

    def test_01(self):
        '''Testing autoflush new disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (chunklen=30 and N=100)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)

        t = bcolz.open(rootdir=self.rootdir)
        assert_array_equal(a, t[:], 'not working correctly')

    def test_02(self):
        '''Testing autoflush when appending data to a disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)

        y = np.array([(-1.0,-2.0)], dtype=[('f0', 'i4'), ('f1', 'f8')])
        t.append(y)

        t = bcolz.open(rootdir=self.rootdir)
        assert_array_equal(np.append(a, y, 0), t[:], 'not working correctly')

    def test_03(self):
        '''Testing autoflush adding column data to a disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)

        y = np.fromiter(((i * 3) for i in xrange(N)), dtype='i8')
        t.addcol(y)

        t = bcolz.open(rootdir=self.rootdir)
        assert_array_equal(a['f0'], t['f0'], 'not working correctly')
        assert_array_equal(a['f1'], t['f1'], 'not working correctly')
        assert_array_equal(y, t['f2'], 'not working correctly')

    def test_04(self):
        '''Testing autoflush deleting a column to a disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)

        y = np.fromiter(((i * 3) for i in xrange(N)), dtype='i8')
        t.delcol('f1')

        t = bcolz.open(rootdir=self.rootdir)
        assert_array_equal(a['f0'], t['f0'], 'not working correctly')
        self.assertTrue('f1' not in t.cols.names)

    def test_05(self):
        '''Testing flush call on a new disk-based ctable'''
        tmp_var = bcolz.ctable.flush
        bcolz.ctable.flush = Mock()
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (chunklen=30 and N=100)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)

        t.flush.assert_called_with()
        bcolz.ctable.flush = tmp_var

    def test_06(self):
        '''Testing flush call when appending data disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)
        t.flush = Mock()

        y = np.array([(-1.0,-2.0)], dtype=[('f0', 'i4'), ('f1', 'f8')])
        t.append(y)

        t.flush.assert_called_with()

    def test_07(self):
        '''Testing flush call after adding column data to a disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)
        t.flush = Mock()

        y = np.fromiter(((i * 3) for i in xrange(N)), dtype='i8')
        t.addcol(y)

        t.flush.assert_called_with()

    def test_08(self):
        '''Testing flush call after deleting a column to a disk-based ctable'''
        N = 100
        a = np.fromiter(((i, i * 2.) for i in xrange(N)), dtype='i4,f8')
        # force ctable with leftovers (N is not a multiple of chunklen)
        t = bcolz.ctable(a, chunklen=30, rootdir=self.rootdir)
        t.flush = Mock()

        y = np.fromiter(((i * 3) for i in xrange(N)), dtype='i8')
        t.delcol('f1')

        t.flush.assert_called_with()

    def test_strings(self):
        """Testing that we can add fixed length strings to a ctable"""
        dtype = np.dtype([("a", "|S5"),
                          ("b", np.uint8),
                          ("c", np.int32),
                          ("d", np.float32)])
        t = bcolz.ctable(np.empty(0, dtype=dtype), mode="w")
        t.append(("aaaaa", 23, 34567, 1.2355))
        self.assertTrue(len(t) == 1)
        self.assertTrue(t["a"][0] == b"aaaaa", t["a"][0])

    def test_auto_flush_constructor_keyword_true_memory(self):
        t = bcolz.ctable([np.empty(0, dtype='i8')], auto_flush=True)
        #self.assertTrue(t.auto_flush)
        # attribute will be False, since it is always false for MemCarray.
        self.assertFalse(t.auto_flush)

    def test_auto_flush_constructor_keyword_true_disk(self):
        t = bcolz.ctable([np.empty(0, dtype='i8')],
                         rootdir=self.rootdir, auto_flush=True)
        self.assertTrue(t.auto_flush)

    def test_auto_flush_constructor_keyword_false_memory(self):
        t = bcolz.ctable([np.empty(0, dtype='i8')], auto_flush=False)
        self.assertFalse(t.auto_flush)

    def test_auto_flush_constructor_keyword_false_disk(self):
        t = bcolz.ctable([np.empty(0, dtype='i8')],
                         rootdir=self.rootdir, auto_flush=False)
        self.assertFalse(t.auto_flush)

    def test_repr_after_appending(self):
        data = np.array([('data1', 1), ('data2', 2), ('data3', 3)],
                        dtype=[('a', 'object'), ('b', 'int64')])
        t = bcolz.ctable(data[:0].copy())
        t.append(data)
        self.assertTrue(bool(repr(t)))

    def test_slice_after_appending(self):
        data = np.array([('data1', 1), ('data2', 2), ('data3', 3)],
                        dtype=[('a', 'object'), ('b', 'int64')])
        t = bcolz.ctable(data[:0].copy())
        t.append(data)
        self.assertTrue((t[:1] == data[:1]).all())


class ContextManagerTest(MayBeDiskTest, TestCase):
    disk = True

    def test_with_statement_flushes(self):

        with bcolz.ctable(np.empty(0, dtype="S2,i4,i8,f8"),
                          rootdir=self.rootdir, mode="w") as x:
            x.append(("a", 1, 2, 3.0))
            x.append(("b", 4, 5, 6.0))

        received = bcolz.ctable(rootdir=self.rootdir)[:]
        expected = \
            np.array([('a', 1, 2, 3.0), ('b', 4, 5, 6.0)],
                     dtype=[('f0', 'S2'), ('f1', '<i4'),
                            ('f2', '<i8'), ('f3', '<f8')])

        assert_array_equal(expected, received)

class ImportTest(TestCase):

     @SkipTest 
     def test_fork(self):
            
        if os.name == 'posix':
            pid = os.fork()
            # terminate nose on child process
            if not pid:
                with self.assertRaises(SystemExit):
                    exit()

if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# tab-width: 4
# fill-column: 72
# End:
