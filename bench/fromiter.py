# Benchmark for assessing the `fromiter()` speed.

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import carray as ca
import itertools as it
from time import time

N = int(1e7)  # the number of elements in x
clevel = 2    # the compression level

print "Creating inputs with %d elements..." % N

x = xrange(N)
y = xrange(1,N+1)
z = xrange(2,N+2)
#ct = ca.ctable((x, y, z), names=['x','y','z'])
#t = ct[:]

print "Starting benchmark now for creating arrays..."
# Create a ndarray
t0 = time()
out = np.fromiter(x, dtype='f8', count=N)
print "Time for array--> %.3f" % (time()-t0,)
print "out-->", out

#ca.set_num_threads(ca.ncores//2)

# Create a carray
t0 = time()
cout = ca.fromiter(iter(x), dtype='f8', count=N, cparms=ca.cparms(clevel))
print "Time for carray--> %.3f" % (time()-t0,)
print "cout-->", cout
#assert_array_equal(out, cout, "Arrays are not equal")

# # Retrieve from a structured ndarray
# t0 = time()
# vals = np.fromiter(iter(t), dtype=t.dtype, count=N)
# print "Time for structured array--> %.3f" % (time()-t0,)
# print "vals-->", len(vals)

# # Retrieve from a ctable
# t0 = time()
# cvals = ca.fromiter(iter(t), dtype=t.dtype, count=N)
# print "Time for ctable--> %.3f" % (time()-t0,)
# print "vals-->", len(cvals)
# assert vals == cvals
