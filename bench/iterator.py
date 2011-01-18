# Benchmark for iterators

import numpy as np
import carray as ca
from time import time

N = 1e7       # the number of elements in x
clevel = 5    # the compression level
sexpr = "(x-1) < 10."  # the expression to compute
#sexpr = "((x-1) % 1000) == 0."  # the expression to compute
#sexpr = "(2*x**3+.3*y**2+z+1)<0"  # the expression to compute

cparams = ca.cparams(clevel)

print "Creating inputs..."

x = np.arange(N)
cx = ca.carray(x, cparams=cparams)
if 'y' not in sexpr:
    ct = ca.ctable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = ca.carray(y, cparams=cparams)
    cz = ca.carray(z, cparams=cparams)
    ct = ca.ctable((cx, cy, cz), names=['x','y','z'])

print "Evaluating...", sexpr
t0 = time()
cbout = ct.eval(sexpr)
print "Time for evaluation--> %.3f" % (time()-t0,)
print "Converting to numy arrays"
bout = cbout[:]
t = ct[:]

t0 = time()
cbool = ca.carray(bout, cparams=cparams)
print "Time for converting boolean--> %.3f" % (time()-t0,)
print "cbool-->", repr(cbool)

t0 = time()
vals = [v for v in cbool.wheretrue()]
print "Time for wheretrue()--> %.3f" % (time()-t0,)
print "vals-->", len(vals)

print "Starting benchmark now..."
# Retrieve from a ndarray
t0 = time()
vals = [v for v in x[bout]]
print "Time for array--> %.3f" % (time()-t0,)
#print "vals-->", len(vals)

#ca.set_num_threads(ca.ncores//2)

# Retrieve from a carray
t0 = time()
#cvals = [v for v in cx[cbout]]
cvals = [v for v in cx.where(cbout)]
print "Time for carray--> %.3f" % (time()-t0,)
#print "vals-->", len(cvals)
assert vals == cvals

# Retrieve from a structured ndarray
t0 = time()
vals = [tuple(v) for v in t[bout]]
print "Time for structured array--> %.3f" % (time()-t0,)
#print "vals-->", len(vals)

# Retrieve from a ctable
t0 = time()
#cvals = [tuple(v) for v in ct[cbout]]
cvals = [v for v in ct.where(cbout)]
print "Time for ctable--> %.3f" % (time()-t0,)
#print "vals-->", len(cvals)
assert vals == cvals
