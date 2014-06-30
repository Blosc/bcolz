from time import time

import numpy as np

import bcolz


N = 1e8
dtype = 'i4'

t0 = time()
a = np.ones(N, dtype=dtype)
print("Time numpy.ones() --> %.4f" % (time() - t0))

t0 = time()
ac = bcolz.fill(N, dtype=dtype, dflt=1)
# ac = bcolz.carray(a)
print("Time carray.fill(dflt=1) --> %.4f" % (time() - t0))

print("ac-->", repr(ac))

t0 = time()
sa = a.sum()
print("Time a.sum() --> %.4f" % (time() - t0))

t0 = time()
sac = ac.sum()
print("Time ac.sum() --> %.4f" % (time() - t0))

assert (sa == sac)
