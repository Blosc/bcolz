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

# Global variable for frame depth testing
GVAR = 1000

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
        """Testing `whereblocks` method with a `outcols` with 2 fields"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', outcols=('f1', 'f2')):
            self.assertEqual(block.dtype.names, ('f1', 'f2'))
            l += len(block)
            s += block['f1'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test03(self):
        """Testing `whereblocks` method with a `outcols` with 1 field"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', outcols=('f1',)):
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

    def test08(self):
        """Testing `whereblocks` method with global and local variables"""
        N = self.N
        lvar = GVAR
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('(f1 + lvar) < (f2 + GVAR)'):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test09(self):
        """Testing `whereblocks` method with vm different than default"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        for block in t.whereblocks('f1 < f2', vm="python"):
            l += len(block)
            s += block['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula


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


class fetchwhereTest(MayBeDiskTest):

    def test00(self):
        """Testing `fetchwhere` method with only an expression"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        ct = t.fetchwhere('f1 < f2')
        l, s = len(ct), ct['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test01(self):
        """Testing `fetchwhere` method with a `outcols` with 2 fields"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        ct = t.fetchwhere('f1 < f2', outcols=('f1', 'f2'))
        self.assertEqual(ct.names, ['f1', 'f2'])
        l, s = len(ct), ct['f1'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test02(self):
        """Testing `fetchwhere` method with a `outcols` with 1 field"""
        N = self.N
        ra = np.fromiter(((i, i, i * 3) for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        ct = t.fetchwhere('f1 < f2', outcols=('f1',))
        self.assertEqual(ct.names, ['f1'])
        l, s = len(ct), ct['f1'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test03(self):
        """Testing `fetchwhere` method with a `limit`, `skip` parameter"""
        N, M = self.N, 101
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        l, s = 0, 0
        ct = t.fetchwhere('f1 < f2', limit=N - M - 2, skip=M)
        l, s = len(ct), ct['f0'].sum()
        self.assertEqual(l, N - M - 2)
        self.assertEqual(s, np.arange(M + 1, N - 1).sum())

    def test04(self):
        """Testing `fetchwhere` method with an `out_flavor` parameter"""
        N = self.N
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        ct = t.fetchwhere('f1 < f2', out_flavor="numpy")
        self.assertEqual(type(ct), np.ndarray)
        l, s = len(ct), ct['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test05(self):
        """Testing `fetchwhere` method with global and local variables"""
        N = self.N
        lvar = GVAR
        ra = np.fromiter(((i, i * 2., i * 3)
                          for i in xrange(N)), dtype='i4,f8,i8')
        t = bcolz.ctable(ra)
        ct = t.fetchwhere('(f1 + lvar) < (f2 + GVAR)', out_flavor="numpy")
        self.assertEqual(type(ct), np.ndarray)
        l, s = len(ct), ct['f0'].sum()
        self.assertEqual(l, N - 1)
        self.assertEqual(s, (N - 1) * (N / 2))  # Gauss summation formula

    def test06(self):
        """Testing `fetchwhere` method off of a timestamp (pd.datetime64)"""
        N = self.N
        query_idx = np.random.randint(0, self.N)
        t = bcolz.fromiter(((i, np.datetime64('2018-03-01') + i) for i in range(N)), dtype="i4,M8[D]", count=N)
        threshold = t[query_idx][1]
        result = t.fetchwhere('(f1 > threshold)', user_dict={'threshold': threshold})
        t_fin = bcolz.fromiter(((i + query_idx, threshold + i) for i in range(1, N - query_idx)), dtype="i4,M8[D]",
                               count=N)
        np.testing.assert_array_equal(result[:], t_fin[:])


class small_fetchwhereTest(fetchwhereTest, TestCase):
    N = 120

class big_fetchwhereTest(fetchwhereTest, TestCase):
    N = 10000


class stringTest(TestCase):

    def test_strings(self):
        """Testing that we can use strings in a variable"""
        dtype = np.dtype([("a", "|S5"),
                          ("b", np.uint8),
                          ("c", np.int32),
                          ("d", np.float32)])
        t = bcolz.ctable(np.empty(0, dtype=dtype))
        strval = b"abcdf"
        t.append(("abcde", 22, 34566, 1.2354))
        t.append((strval, 23, 34567, 1.2355))
        t.append(("abcde", 22, 34566, 1.2354))
        res = list(t.eval('a == strval'))
        self.assertTrue(res == [False, True, False],
                        "querying strings not working correctly")

    def test_strings2(self):
        """Testing that we can use strings in a variable (II)"""
        dtype = np.dtype([("STATE", "|S32"),
                          ("b", np.int32)])
        recarr = np.array([('California', 1), ('Dakota', 9)], dtype=dtype)
        t = bcolz.ctable(recarr)
        res = [tuple(row) for row in t.where(
            "STATE == b'California'", outcols=["nrow__", "b"])]
        self.assertTrue(res == [(0, 1)],
                        "querying strings not working correctly")
        # test with unicode
        res = [tuple(row) for row in t.where(
            u"STATE == b'California'", outcols=["nrow__", "b"])]
        self.assertTrue(res == [(0, 1)],
                        "querying strings not working correctly with unicode querystring")



if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# py-indent-offset: 4
# tab-width: 4
# fill-column: 72
# End:
