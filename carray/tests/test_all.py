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
import carray.tests

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


def test():
    """
    Run all the tests in the test suite.
    """
    print_versions()
    unittest.TextTestRunner().run(suite())


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

    print_versions()
    if not only_versions:
        unittest.main(defaultTest='carray.tests.suite')


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
