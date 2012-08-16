########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################

"""
Run all test cases.
"""

import sys, os
import unittest

import numpy
import carray
from carray.tests import common


# Recommended minimum versions
min_numpy_version = "1.5"


def suite():
    test_modules = [
        'carray.tests.test_carray',
        'carray.tests.test_ctable',
        'carray.tests.test_ndcarray',
        'carray.tests.test_queries',
        ]
    alltests = unittest.TestSuite()
    for name in test_modules:
        exec('from %s import suite as test_suite' % name)
        alltests.addTest(test_suite())
    return alltests


def print_versions():
    """Print all the versions of software that carray relies on."""
    print("-=" * 38)
    print("carray version:    %s" % carray.__version__)
    print("NumPy version:     %s" % numpy.__version__)
    tinfo = carray.blosc_version()
    print("Blosc version:     %s (%s)" % (tinfo[0], tinfo[1]))
    if carray.numexpr_here:
        print("Numexpr version:   %s" % carray.numexpr.__version__)
    else:
        print("Numexpr version:   not available "
              "(version >= %s not detected)" %  carray.min_numexpr_version)
    from Cython.Compiler.Main import Version as Cython_Version
    print("Cython version:    %s" % Cython_Version.version)
    print("Python version:    %s" % sys.version)
    if os.name == "posix":
        (sysname, nodename, release, version, machine) = os.uname()
        print("Platform:          %s-%s" % (sys.platform, machine))
    print("Byte-ordering:     %s" % sys.byteorder)
    print("Detected cores:    %s" % carray.detect_number_of_cores())
    print("-=" * 38)


def print_heavy(heavy):
    if heavy:
        print """\
Performing the complete test suite!"""
    else:
        print """\
Performing only a light (yet comprehensive) subset of the test suite.
If you want a more complete test, try passing the --heavy flag to this script
(or set the 'heavy' parameter in case you are using carray.test() call).
The whole suite will take more than 30 seconds to complete on a relatively
modern CPU and around 100 MB of disk.
"""
    print '-=' * 38

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
        print("*Warning*: NumPy version is lower than recommended: %s < %s" % \
              (numpy.__version__, min_numpy_version))

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
        unittest.main(defaultTest='carray.tests.suite')


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
