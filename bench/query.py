# Benchmark to compare the times for evaluating queries.
# Numexpr is needed in order to execute this.

import math
from time import time

import numpy as np

import bcolz


N = 1e7  # the number of elements in x
clevel = 5  # the compression level
cname = "blosclz"  # the compressor name
sexpr = "(x+1)<10"  # small number of items
# sexpr = "(x+1)<1000000"              # large number
sexpr = "(2*x*x*x+.3*y**2+z+1)<10"  # small number
#sexpr = "(2*x*x*x+.3*y**2+z+1)<1e15"  # medium number
#sexpr = "(2*x*x*x+.3*y**2+z+1)<1e20"  # large number

print("Creating inputs...")

cparams = bcolz.cparams(clevel=clevel, cname=cname)

x = np.arange(N)
cx = bcolz.carray(x, cparams=cparams)
if 'y' not in sexpr:
    t = bcolz.ctable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = bcolz.carray(y, cparams=cparams)
    cz = bcolz.carray(z, cparams=cparams)
    t = bcolz.ctable((cx, cy, cz), names=['x', 'y', 'z'])
nt = t[:]

print("Querying '%s' with 10^%d points" % (sexpr, int(math.log10(N))))

t0 = time()
out = [r for r in x[eval(sexpr)]]
print("Time for numpy--> %.3f" % (time() - t0,))

t0 = time()
out = [r for r in t[eval(sexpr)]]
print("Time for structured array--> %.3f" % (time() - t0,))

t0 = time()
out = [r for r in cx[sexpr]]
print("Time for carray --> %.3f" % (time() - t0,))

# Uncomment the next for disabling threading
#ne.set_num_threads(1)
#bcolz.blosc_set_num_threads(1)
# Seems that this works better if we dividw the number of cores by 2.
# Maybe due to some contention between Numexpr and Blosc?
#bcolz.set_num_threads(bcolz.ncores//2)

t0 = time()
#cout = t[t.eval(sexpr, cparams=cparams)]
cout = [r for r in t.where(sexpr)]
#cout = [r['x'] for r in t.where(sexpr)]
#cout = [r['y'] for r in t.where(sexpr, colnames=['x', 'y'])]
print("Time for ctable--> %.3f" % (time() - t0,))
print("cout-->", len(cout), cout[:10])

#assert_array_equal(out, cout, "Arrays are not equal")
