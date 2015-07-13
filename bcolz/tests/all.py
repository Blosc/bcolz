########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""
Run all test cases.
"""

import sys
import os

import numpy
import bcolz
from bcolz.tests import common
from bcolz.tests.common import unittest


# Recommended minimum versions
min_numpy_version = "1.7"


def suite():
    this_dir = os.path.dirname(__file__)
    return unittest.TestLoader().discover(
        start_dir=this_dir, pattern="test_*.py")


def print_heavy(heavy):
    if heavy:
        print("""\
Performing the complete test suite!""")
    else:
        print("""\
Performing only a light (yet comprehensive) subset of the test suite.
If you want a more complete test, try passing the '-heavy' flag to this
script (or set the 'heavy' parameter in case you are using bcolz.test()
call).  The whole suite will take more than 30 seconds to complete on a
relatively modern CPU and around 300 MB of RAM and 500 MB of disk
[32-bit platforms will always run significantly more lightly].
""")
    print('-=' * 38)


def test(verbose=False, heavy=False):
    """
    test(verbose=False, heavy=False)

    Run all the tests in the test suite.

    If `verbose` is set, the test suite will emit messages with full
    verbosity (not recommended unless you are looking into a certain
    problem).

    If `heavy` is set, the test suite will be run in *heavy* mode (you
    should be careful with this because it can take a lot of time and
    resources from your computer).
    """
    bcolz.print_versions()
    print_heavy(heavy)

    # What a context this is!
    oldverbose, common.verbose = common.verbose, verbose
    oldheavy, common.heavy = common.heavy, heavy
    try:
        ret = unittest.TextTestRunner().run(suite())
        sys.exit(ret.wasSuccessful() == False)
    finally:
        common.verbose = oldverbose
        common.heavy = oldheavy  # there are pretty young heavies, too ;)

if __name__ == '__main__':

    if numpy.__version__ < min_numpy_version:
        print("*Warning*: NumPy version is lower than recommended:"
              "%s < %s" % (numpy.__version__, min_numpy_version))

    # Handle some global flags (i.e. only useful for test_all.py)
    only_versions = 0
    args = sys.argv[:]
    for arg in args:
        if arg in ['-print-versions']:
            only_versions = True
            sys.argv.remove(arg)
        if arg in ['-verbose']:
            common.verbose = True
            sys.argv.remove(arg)
        if arg in ['-heavy']:
            common.heavy = True
            sys.argv.remove(arg)

    bcolz.print_versions()
    if not only_versions:
        print_heavy(common.heavy)
        unittest.TextTestRunner().run(suite())


# Local Variables:
# mode: python
# fill-column: 72
# End:
