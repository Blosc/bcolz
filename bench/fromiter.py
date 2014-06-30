# Benchmark for assessing the `fromiter()` speed.

from time import time

import numpy as np
from numpy.testing import assert_array_equal

import bcolz
from bcolz.py2help import xrange, izip


N = int(1e6)  # the number of elements in x
clevel = 2  # the compression level

print("Creating inputs with %d elements..." % N)

x = xrange(N)  # not a true iterable, but can be converted
y = xrange(1, N + 1)
z = xrange(2, N + 2)

print("Starting benchmark now for creating arrays...")
# Create a ndarray
# x = (i for i in xrange(N))    # true iterable
t0 = time()
out = np.fromiter(x, dtype='f8', count=N)
print("Time for array--> %.3f" % (time() - t0,))
print("out-->", len(out))

#bcolz.set_num_threads(bcolz.ncores//2)

# Create a carray
#x = (i for i in xrange(N))    # true iterable
t0 = time()
cout = bcolz.fromiter(x, dtype='f8', count=N, cparams=bcolz.cparams(clevel))
print("Time for carray--> %.3f" % (time() - t0,))
print("cout-->", len(cout))
assert_array_equal(out, cout, "Arrays are not equal")

# Create a carray (with unknown size)
#x = (i for i in xrange(N))    # true iterable
t0 = time()
cout = bcolz.fromiter(x, dtype='f8', count=-1, cparams=bcolz.cparams(clevel))
print("Time for carray (count=-1)--> %.3f" % (time() - t0,))
print("cout-->", len(cout))
assert_array_equal(out, cout, "Arrays are not equal")

# Retrieve from a structured ndarray
gen = ((i, j, k) for i, j, k in izip(x, y, z))
t0 = time()
out = np.fromiter(gen, dtype="f8,f8,f8", count=N)
print("Time for structured array--> %.3f" % (time() - t0,))
print("out-->", len(out))

# Retrieve from a ctable
gen = ((i, j, k) for i, j, k in izip(x, y, z))
t0 = time()
cout = bcolz.fromiter(gen, dtype="f8,f8,f8", count=N)
print("Time for ctable--> %.3f" % (time() - t0,))
print("out-->", len(cout))
assert_array_equal(out, cout[:], "Arrays are not equal")

# Retrieve from a ctable (with unknown size)
gen = ((i, j, k) for i, j, k in izip(x, y, z))
t0 = time()
cout = bcolz.fromiter(gen, dtype="f8,f8,f8", count=-1)
print("Time for ctable (count=-1)--> %.3f" % (time() - t0,))
print("out-->", len(cout))
assert_array_equal(out, cout[:], "Arrays are not equal")
