########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""
bcolz: columnar and compressed data containers
==============================================

bcolz provides columnar and compressed data containers.  Column storage
allows for efficiently querying tables with a large number of columns.  It
also allows for cheap addition and removal of column.  In addition,
bcolz objects are compressed by default for reducing memory/disk I/O needs.
The compression process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

"""

min_numexpr_version = '1.4.1'  # the minimum version of Numexpr needed
numexpr_here = False
try:
    import numexpr
except ImportError:
    pass
else:
    if numexpr.__version__ >= min_numexpr_version:
        numexpr_here = True

# Check for pandas (for data container conversion purposes)
pandas_here = False
try:
    import pandas
except ImportError:
    pass
else:
    pandas_here = True

# Check for PyTables (for data container conversion purposes)
tables_here = False
try:
    import tables
except ImportError:
    pass
else:
    tables_here = True


# Print array functions (imported from NumPy)
from bcolz.arrayprint import (
    array2string, set_printoptions, get_printoptions )

from bcolz.carray_ext import (
    carray, blosc_version, blosc_compressor_list,
    _blosc_set_nthreads as blosc_set_nthreads,
    _blosc_init, _blosc_destroy)
from bcolz.ctable import ctable
from bcolz.toplevel import (
    print_versions, detect_number_of_cores, set_nthreads,
    open, fromiter, arange, zeros, ones, fill,
    iterblocks, cparams, walk)
from bcolz.chunked_eval import eval
from bcolz.defaults import defaults
from bcolz.version import version as __version__

try:
    from bcolz.tests import test
except ImportError:
    def test(*args, **kwargs):
        print("Could not import tests.\n"
              "If on Python2.6 please install unittest2")


def _get_git_descrtiption(path_):
    """ Get the output of git-describe when executed in a given path. """

    # imports in function because:
    # a) easier to refactor
    # b) clear they are only used here
    import subprocess
    import os
    import os.path as path
    from bcolz.py2help import check_output

    # make an absolute path if required, for example when running in a clone
    if not path.isabs(path_):
        path_ = path.join(os.getcwd(), path_)
    # look up the commit using subprocess and git describe
    try:
        # redirect stderr to stdout to make sure the git error message in case
        # we are not in a git repo doesn't appear on the screen and confuse the
        # user.
        label = check_output(["git", "describe"], cwd=path_,
                             stderr=subprocess.STDOUT).strip()
        return label
    except OSError:  # in case git wasn't found
        pass
    except subprocess.CalledProcessError:  # not in git repo
        pass

git_description = _get_git_descrtiption(__path__[0])

# Initialization code for the Blosc and numexpr libraries
_blosc_init()
ncores = detect_number_of_cores()
blosc_set_nthreads(ncores)
# Benchmarks show that using several threads can be an advantage in bcolz
blosc_set_nthreads(ncores)
if numexpr_here:
    numexpr.set_num_threads(ncores)
import atexit
atexit.register(_blosc_destroy)
