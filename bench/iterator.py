# Benchmark for iterators

from time import time

import numpy as np

import bcolz


N = 1e8  # the number of elements in x
clevel = 5  # the compression level
sexpr = "(x-1) < 10."  # the expression to compute
# sexpr = "((x-1) % 1000) == 0."  # the expression to compute
#sexpr = "(2*x**3+.3*y**2+z+1)<0"  # the expression to compute

cparams = bcolz.cparams(clevel)

print("Creating inputs...")

x = np.arange(N)
cx = bcolz.carray(x, cparams=cparams)
if 'y' not in sexpr:
    ct = bcolz.ctable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = bcolz.carray(y, cparams=cparams)
    cz = bcolz.carray(z, cparams=cparams)
    ct = bcolz.ctable((cx, cy, cz), names=['x', 'y', 'z'])

print("Evaluating...", sexpr)
t0 = time()
cbout = ct.eval(sexpr)
print("Time for evaluation--> %.3f" % (time() - t0,))
print("Converting to numy arrays")
bout = cbout[:]
t = ct[:]

t0 = time()
cbool = bcolz.carray(bout, cparams=cparams)
print("Time for converting boolean--> %.3f" % (time() - t0,))
print("cbool-->", repr(cbool))

t0 = time()
vals = [v for v in cbool.wheretrue()]
print("Time for wheretrue()--> %.3f" % (time() - t0,))
print("vals-->", len(vals))

print("Starting benchmark now...")
# Retrieve from a ndarray
t0 = time()
vals = [v for v in x[bout]]
print("Time for array--> %.3f" % (time() - t0,))
#print("vals-->", len(vals))

#bcolz.set_num_threads(bcolz.ncores//2)

# Retrieve from a carray
t0 = time()
#cvals = [v for v in cx[cbout]]
cvals = [v for v in cx.where(cbout)]
print("Time for carray--> %.3f" % (time() - t0,))
#print("vals-->", len(cvals))
assert vals == cvals

# Retrieve from a structured ndarray
t0 = time()
vals = [tuple(v) for v in t[bout]]
print("Time for structured array--> %.3f" % (time() - t0,))
#print("vals-->", len(vals))

# Retrieve from a ctable
t0 = time()
#cvals = [tuple(v) for v in ct[cbout]]
cvals = [v for v in ct.where(cbout)]
print("Time for ctable--> %.3f" % (time() - t0,))
#print("vals-->", len(cvals))
assert vals == cvals
