########################################################################
#
#       License: BSD
#       Created: August 5, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: __init__.py 4463 2010-06-04 15:17:09Z faltet $
#
########################################################################

"""
Unit tests for carray
=====================

This package contains some modules which provide a ``suite()``
function (with no arguments) which returns a test suite for some
carray functionality.
"""

from carray.tests.test_all import print_versions, test, suite
