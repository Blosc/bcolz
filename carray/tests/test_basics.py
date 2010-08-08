########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: test_basics.py 4463 2010-06-04 15:17:09Z faltet $
#
########################################################################

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
import unittest


class BasicCheck(unittest.TestCase):
    
    def test00(self):
        #a = np.arange(1e4)
        a = np.linspace(-1, 1, 1e4)
        b = ca.carray(a)
        print "b->", `b`
        c = b.toarray()
        print "c->", `c`
        assert_array_equal(a, c)


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(BasicCheck))

    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
