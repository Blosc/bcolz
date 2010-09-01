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


class basicTest(unittest.TestCase):

    def test00a(self):
        """Testing ctable creation from a tuple of carrays"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'))
        #print "t->", `t`
        #print "t[:]", t[:]
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing ctable creation from a tuple of numpy arrays"""
        N = 1e1
        a = np.arange(N, dtype='i4')
        b = np.arange(N, dtype='f8')+1
        t = ca.ctable((a, b), ('f0', 'f1'))
        #print "t->", `t`
        #print "t[:]", t[:]
        ra = np.rec.fromarrays([a,b]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable creation from an structured array"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        t = ca.ctable(ra)
        #print "t->", `t`
        #print "t[:]", t[:]
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")



def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(basicTest))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
