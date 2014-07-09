########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - francesc@blosc.io
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
from bcolz.tests.common import (
    MayBeDiskTest, TestCase, unittest, skipUnless, SkipTest)


# Recommended minimum versions
min_numpy_version = "1.7"

def suite():
    this_dir = os.path.dirname(__file__)
    return unittest.TestLoader().discover(
        start_dir=this_dir, pattern = "test_*.py")


def print_versions():
    """Print all the versions of software that bcolz relies on."""
    print("-=" * 38)
    print("bcolz version:     %s" % bcolz.__version__)
    print("NumPy version:     %s" % numpy.__version__)
    tinfo = bcolz.blosc_version()
    blosc_cnames = bcolz.blosc_compressor_list()
    print("Blosc version:     %s (%s)" % (tinfo[0], tinfo[1]))
    print("Blosc compressors: %s" % (blosc_cnames,))
    if bcolz.numexpr_here:
        print("Numexpr version:   %s" % bcolz.numexpr.__version__)
    else:
        print("Numexpr version:   not available "
              "(version >= %s not detected)" %  bcolz.min_numexpr_version)
    print("Python version:    %s" % sys.version)
    if os.name == "posix":
        (sysname, nodename, release, version, machine) = os.uname()
        print("Platform:          %s-%s" % (sys.platform, machine))
    print("Byte-ordering:     %s" % sys.byteorder)
    print("Detected cores:    %s" % bcolz.detect_number_of_cores())
    print("-=" * 38)


def print_heavy(heavy):
    if heavy:
        print("""\
Performing the complete test suite!""")
    else:
        print("""\
Performing only a light (yet comprehensive) subset of the test suite.
If you want a more complete test, try passing the --heavy flag to this
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
    print_versions()
    print_heavy(heavy)

    # What a context this is!
    oldverbose, common.verbose = common.verbose, verbose
    oldheavy, common.heavy = common.heavy, heavy
    try:
        unittest.TextTestRunner().run(suite())
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
        if arg in ['--print-versions']:
            only_versions = True
            sys.argv.remove(arg)
        if arg in ['--verbose']:
            common.verbose = True
            sys.argv.remove(arg)
        if arg in ['--heavy']:
            common.heavy = True
            sys.argv.remove(arg)

    print_versions()
    if not only_versions:
        print_heavy(common.heavy)
        unittest.main(defaultTest='bcolz.tests.suite')


## Local Variables:
## mode: python
## fill-column: 72
## End:
