# Benchmark that compares the times for concatenating arrays with
# compressed arrays vs plain numpy arrays.  The 'numpy' and 'concat'
# styles are for regular numpy arrays, while 'carray' is for carrays.
#
# Call this benchmark as:
#
# python bench/concat.py style
#
# where `style` can be any of 'numpy', 'concat' or 'bcolsz'
#
# You can modify other parameters from the command line if you want:
#
# python bench/concat.py style arraysize nchunks nrepeats clevel
#

from __future__ import absolute_import

import sys
import math
import time

import numpy

import bcolz
from bcolz.py2help import xrange
from .bench_helper import ctime


def concat(data):
    tlen = sum(x.shape[0] for x in data)
    alldata = numpy.empty((tlen,))
    pos = 0
    for x in data:
        step = x.shape[0]
        alldata[pos:pos + step] = x
        pos += step

    return alldata


def append(data, clevel):
    alldata = bcolz.carray(data[0], cparams=bcolz.cparams(clevel))
    for carr in data[1:]:
        alldata.append(carr)

    return alldata


class Suite:
    a = None
    N = 1000000
    K = 10
    T = 3
    clevel = 1
    style = 'bcolz'
    r = None

    def __init__(self, N=1000000, K=10, T=3, clevel=1, style='bcolz'):
        Suite.N = N
        Suite.K = K
        Suite.T = T
        Suite.clevel = clevel
        Suite.style = style
        Suite.r = None

    def setup(self):
        # The next datasets allow for very high compression ratios
        Suite.a = [numpy.arange(Suite.N, dtype='f8') for _ in range(Suite.K)]
        print("problem size: (%d) x %d = 10^%g" % (Suite.N, Suite.K,
                                                   math.log10(Suite.N * Suite.K)))

    def time_concatenate(self):
        if Suite.style == 'numpy':
            for _ in xrange(Suite.T):
                Suite.r = numpy.concatenate(Suite.a, 0)
        elif Suite.style == 'concat':
            for _ in xrange(Suite.T):
                Suite.r = concat(Suite.a)
        elif Suite.style == 'bcolz':
            for _ in xrange(Suite.T):
                Suite.r = append(Suite.a, Suite.clevel)

    def print_container_size(self):
        if Suite.style == 'bcolz':
            size = Suite.r.cbytes
        else:
            size = Suite.r.size * Suite.r.dtype.itemsize
        print("size of the final container: %.3f MB" %
              (size / float(1024 * 1024)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            "Pass at least one of these styles: 'numpy', 'concat' or 'bcolz' ")
        sys.exit(1)

    style = sys.argv[1]
    if len(sys.argv) == 2:
        N, K, T, clevel = (1000000, 10, 3, 1)
    else:
        N, K, T = [int(arg) for arg in sys.argv[2:5]]
        if len(sys.argv) > 5:
            clevel = int(sys.argv[5])
        else:
            clevel = 0

    # run benchmark
    suite = Suite(N=N, K=K, T=T, clevel=clevel, style=style)
    suite.setup()
    with ctime("time_concatenate"):
        suite.time_concatenate()
    suite.print_container_size()
