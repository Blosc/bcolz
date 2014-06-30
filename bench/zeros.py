from time import time

import numpy as np

import bcolz


N = 2e8
dtype = 'i4'

t0 = time()
a = np.zeros(N, dtype=dtype)
print("Time numpy.zeros() --> %.4f" % (time() - t0))

t0 = time()
ac = bcolz.zeros(N, dtype=dtype)
# ac = bcolz.carray(a)
print("Time bcolz.zeros() --> %.4f" % (time() - t0))

print("ac-->", repr(ac))

#assert(np.all(a == ac))
