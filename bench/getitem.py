# Benchmark for getitem

import numpy as np
import carray as ca
from time import time

N = 1e7       # the number of elements in x
M = 100000    # the elements to get
clevel = 1    # the compression level

print "Creating inputs with %d elements..." % N

cparams = ca.cparams(clevel)

x = np.arange(N)
y = np.arange(N)
z = np.arange(N)
cx = ca.carray(x, cparams=cparams)
cy = ca.carray(y, cparams=cparams)
cz = ca.carray(z, cparams=cparams)
ct = ca.ctable((cx, cy, cz), names=['x','y','z'])
t = ct[:]

print "Starting benchmark now for getting %d elements..." % M
# Retrieve from a ndarray
t0 = time()
vals = [x[i] for i in xrange(0, M, 3)]
print "Time for array--> %.3f" % (time()-t0,)
print "vals-->", len(vals)

#ca.set_num_threads(ca.ncores//2)

# Retrieve from a carray
t0 = time()
cvals = [cx[i] for i in xrange(0, M, 3)]
#cvals = cx[:M:3][:].tolist()
print "Time for carray--> %.3f" % (time()-t0,)
print "vals-->", len(cvals)
assert vals == cvals

# Retrieve from a structured ndarray
t0 = time()
vals = [t[i] for i in xrange(0, M, 3)]
print "Time for structured array--> %.3f" % (time()-t0,)
print "vals-->", len(vals)

# Retrieve from a ctable
t0 = time()
cvals = [ct[i] for i in xrange(0, M, 3)]
#cvals = ct[:M:3][:].tolist()
print "Time for ctable--> %.3f" % (time()-t0,)
print "vals-->", len(cvals)
assert vals == cvals
