from time import time

import numpy as np

import bcolz


N = 1e8
dtype = 'i4'

start, stop, step = 5, N, 4

t0 = time()
a = np.arange(start, stop, step, dtype=dtype)
print("Time numpy.arange() --> %.3f" % (time() - t0))

t0 = time()
ac = bcolz.arange(start, stop, step, dtype=dtype)
print("Time bcolsz.arange() --> %.3f" % (time() - t0))

print("ac-->", repr(ac))

# assert(np.all(a == ac))
