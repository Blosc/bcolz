########################################################################
#
#       License: BSD
#       Created: January 18, 2011
#       Author:  Francesc Alted - francesc@blosc.io
#
########################################################################

from __future__ import absolute_import

import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import bcolz
from bcolz.tests import common
from bcolz.tests.common import (
    MayBeDiskTest, TestCase, unittest, skipUnless)


class with_listTest():

    def test00a(self):
        """Testing wheretrue() in combination with a list constructor"""
        a = bcolz.zeros(self.N, dtype="bool")
        a[30:40] = bcolz.ones(10, dtype="bool")
        alist = list(a)
        blist1 = [r for r in a.wheretrue()]
        self.assertTrue(blist1 == list(range(30,40)))
        alist2 = list(a)
        self.assertTrue(alist == alist2, "wheretrue() not working correctly")

    def test00b(self):
        """Testing wheretrue() with a multidimensional array"""
        a = bcolz.zeros((self.N, 10), dtype="bool")
        a[30:40] = bcolz.ones(10, dtype="bool")
        self.assertRaises(NotImplementedError, a.wheretrue)

    def test01a(self):
        """Testing where() in combination with a list constructor"""
        a = bcolz.zeros(self.N, dtype="bool")
        a[30:40] = bcolz.ones(10, dtype="bool")
        b = bcolz.arange(self.N, dtype="f4")
        blist = list(b)
        blist1 = [r for r in b.where(a)]
        self.assertTrue(blist1 == list(range(30,40)))
        blist2 = list(b)
        self.assertTrue(blist == blist2, "where() not working correctly")

    def test01b(self):
        """Testing where() with a multidimensional array"""
        a = bcolz.zeros((self.N, 10), dtype="bool")
        a[30:40] = bcolz.ones(10, dtype="bool")
        b = bcolz.arange(self.N*10, dtype="f4").reshape((self.N, 10))
        self.assertRaises(NotImplementedError, b.where, a)

    def test02(self):
        """Testing iter() in combination with a list constructor"""
        b = bcolz.arange(self.N, dtype="f4")
        blist = list(b)
        blist1 = [r for r in b.iter(3,10)]
        self.assertTrue(blist1 == list(range(3,10)))
        blist2 = list(b)
        self.assertTrue(blist == blist2, "iter() not working correctly")


class small_with_listTest(with_listTest, TestCase):
    N = 100

class big_with_listTest(with_listTest, TestCase):
    N = 10000


if __name__ == '__main__':
    unittest.main(verbosity=2)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
