########################################################################
#
#       License: BSD
#       Created: January 18, 2011
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import


import numpy as np
import bcolz
from bcolz.py2help import xrange
from bcolz.tests.common import (
    MayBeDiskTest, TestCase, unittest)


class listTest(MayBeDiskTest):

    def test00a(self):
        """Testing wheretrue() in combination with a list constructor"""
        a = bcolz.zeros(self.N, dtype="bool", rootdir=self.rootdir)
        a[30:40] = bcolz.ones(10, dtype="bool")
        alist = list(a)
        blist1 = [r for r in a.wheretrue()]
        self.assertTrue(blist1 == list(range(30, 40)))
        alist2 = list(a)
        self.assertTrue(alist == alist2, "wheretrue() not working correctly")

    def test00b(self):
        """Testing wheretrue() with a multidimensional array"""
        a = bcolz.zeros((self.N, 10), dtype="bool", rootdir=self.rootdir)
        a[30:40] = bcolz.ones(10, dtype="bool")
        self.assertRaises(NotImplementedError, a.wheretrue)

    def test01a(self):
        """Testing where() in combination with a list constructor"""
        a = bcolz.zeros(self.N, dtype="bool", rootdir=self.rootdir)
        a[30:40] = bcolz.ones(10, dtype="bool")
        b = bcolz.arange(self.N, dtype="f4")
        blist = list(b)
        blist1 = [r for r in b.where(a)]
        self.assertTrue(blist1 == list(range(30, 40)))
        blist2 = list(b)
        self.assertTrue(blist == blist2, "where() not working correctly")

    def test01b(self):
        """Testing where() with a multidimensional array"""
        a = bcolz.zeros((self.N, 10), dtype="bool", rootdir=self.rootdir)
        a[30:40] = bcolz.ones(10, dtype="bool")
        b = bcolz.arange(self.N * 10, dtype="f4").reshape((self.N, 10))
        self.assertRaises(NotImplementedError, b.where, a)

    def test02(self):
        """Testing iter() in combination with a list constructor"""
        b = bcolz.arange(self.N, dtype="f4", rootdir=self.rootdir)
        blist = list(b)
        blist1 = [r for r in b.iter(3, 10)]
        self.assertTrue(blist1 == list(range(3, 10)))
        blist2 = list(b)
        self.assertTrue(blist == blist2, "iter() not working correctly")


class small_listTest(listTest, TestCase):
    N = 100


class big_listTest(listTest, TestCase):
    N = 10000


class small_listDiskTest(listTest, TestCase):
    N = 100
    disk = True


class big_listDiskTest(listTest, TestCase):
    N = 10000
    disk = True


class whereblocksTest(MayBeDiskTest):

    def test00(self):
        """Testing `whereblocks` method with only an expression"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2'):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test01(self):
        """Testing `whereblocks` method with a `blen`"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f0 <= f1', blen=100):
            l += len(block)
            # All blocks should be of length 100, except the last one,
            # which should be 0 or 20
            self.assertTrue(len(block) in (0, 20, 100))
            s += block['f0'].sum()
        self.assertEqual(l, N)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test02(self):
        """Testing `whereblocks` method with a `outfields` with 2 fields"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', outfields=('f1', 'f2')):
            self.assertEqual(block.dtype.names, ('f1', 'f2'))
            l += len(block)
            s += block['f1'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test03(self):
        """Testing `whereblocks` method with a `outfields` with 1 field"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', outfields=('f1',)):
            self.assertEqual(block.dtype.names, ('f1',))
            l += len(block)
            s += block['f1'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test04(self):
        """Testing `whereblocks` method with a `limit` parameter"""
        N, M = self.N, 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', limit=M):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, M)
        self.assertEqual(s, M * ((M + 1) / 2))  # Gauss summation formula

    def test05(self):
        """Testing `whereblocks` method with a `limit` parameter"""
        N, M = self.N, 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', limit=M):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, M)
        self.assertEqual(s, M * ((M + 1) / 2))  # Gauss summation formula

    def test06(self):
        """Testing `whereblocks` method with a `skip` parameter"""
        N, M = self.N, 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', skip=N - M):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, M - 1)
        self.assertEqual(s, np.arange(N - M + 1, N).sum())

    def test07(self):
        """Testing `whereblocks` method with a `limit`, `skip` parameter"""
        N, M = self.N, 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', limit=N - M - 2, skip=M):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N - M - 2)
        self.assertEqual(s, np.arange(M + 1, N - 1).sum())


class small_whereblocksTest(whereblocksTest, TestCase):
    N = 120


class big_whereblocksTest(whereblocksTest, TestCase):
    N = 10000


class small_whereblocksDiskTest(whereblocksTest, TestCase):
    N = 120
    disk = True


class big_whereblocksDiskTest(whereblocksTest, TestCase):
    N = 10000
    disk = True


if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# py-indent-offset: 4
# tab-width: 4
# fill-column: 72
# End:
