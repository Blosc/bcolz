# Benchmark to compare times for iterators in generator contexts by
# using carrays vs plain numpy arrays.

from time import time

import numpy as np

import bcolz


N = 1e6

a = np.arange(N)
b = bcolz.carray(a)

t0 = time()
# sum1 = sum(a)
sum1 = sum((v for v in a[2::3] if v < 10))
t1 = time() - t0
print("Summing using numpy iterator: %.3f" % t1)

t0 = time()
#sum2 = sum(b)
sum2 = sum((v for v in b.iter(2, None, 3) if v < 10))
t2 = time() - t0
print("Summing using carray iterator: %.3f  speedup: %.2f" % (t2, t1 / t2))

assert sum1 == sum2, "Summations are not equal!"
