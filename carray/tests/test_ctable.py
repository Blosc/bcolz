########################################################################
#
#       License: BSD
#       Created: September 1, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: test_ctable.py $
#
########################################################################

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


class add_del_colTest(unittest.TestCase):

    def test00(self):
        """Testing appending a new column (carray flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(ca.carray(c), 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing appending a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing appending a new column (default naming)"""
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


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(createTest))
    theSuite.addTest(unittest.makeSuite(add_del_colTest))
    theSuite.addTest(unittest.makeSuite(getitemTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
