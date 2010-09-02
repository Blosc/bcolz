########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: test_all.py 4463 2010-06-04 15:17:09Z faltet $
#
########################################################################

"""
Run all test cases.
"""

import sys, os
import unittest

import numpy
import carray
import carray.tests
from carray.utils import detectNumberOfCores

# Recommended minimum versions
min_numpy_version = "1.3"


def suite():
    test_modules = [
        'carray.tests.test_carray',
        'carray.tests.test_ctable',
        ]
    alltests = unittest.TestSuite()
    for name in test_modules:
        exec('from %s import suite as test_suite' % name)
        alltests.addTest(test_suite())
    return alltests


def print_versions():
    """Print all the versions of software that carray relies on."""
    print '-=' * 38
    print "carray version:    %s" % carray.__version__
    print "NumPy version:     %s" % numpy.__version__
    tinfo = carray.whichLibVersion("blosc")
    if tinfo is not None:
        print "Blosc version:     %s (%s)" % (tinfo[0], tinfo[1])
    from Cython.Compiler.Main import Version as Cython_Version
    print 'Cython version:    %s' % Cython_Version.version
    print 'Python version:    %s' % sys.version
    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print 'Platform:          %s-%s' % (sys.platform, machine)
    print 'Byte-ordering:     %s' % sys.byteorder
    print 'Detected cores:    %s' % detectNumberOfCores()
    print '-=' * 38


def test(verbose=False, heavy=False):
    """
    Run all the tests in the test suite.

    If `verbose` is set, the test suite will emit messages with full
    verbosity (not recommended unless you are looking into a certain
    problem).
    """
    print_versions()

    # What a context this is!
    oldverbose, common.verbose = common.verbose, verbose
    try:
        unittest.TextTestRunner().run(suite())
    finally:
        common.verbose = oldverbose


if __name__ == '__main__':

    if numpy.__version__ < min_numpy_version:
        print "*Warning*: NumPy version is lower than recommended: %s < %s" % \
              (numpy.__version__, min_numpy_version)

    # Handle some global flags (i.e. only useful for test_all.py)
    only_versions = 0
    args = sys.argv[:]
    for arg in args:
        if arg in ['--print-versions']:
            only_versions = True
            sys.argv.remove(arg)

    print_versions()
    if not only_versions:
        unittest.main(defaultTest='carray.tests.suite')


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
