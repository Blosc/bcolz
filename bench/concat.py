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


if len(sys.argv) < 2:
    print("Pass at least one of these styles: 'numpy', 'concat' or 'bcolz' ")
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

# The next datasets allow for very high compression ratios
a = [numpy.arange(N, dtype='f8') for _ in range(K)]
print("problem size: (%d) x %d = 10^%g" % (N, K, math.log10(N * K)))

t = time.time()
if style == 'numpy':
    for _ in xrange(T):
        r = numpy.concatenate(a, 0)
elif style == 'concat':
    for _ in xrange(T):
        r = concat(a)
elif style == 'bcolz':
    for _ in xrange(T):
        r = append(a, clevel)

t = time.time() - t
print('time for concat: %.3fs' % (t / T))

if style == 'bcolz':
    size = r.cbytes
else:
    size = r.size * r.dtype.itemsize
print("size of the final container: %.3f MB" % (size / float(1024 * 1024)) )
