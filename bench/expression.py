# Benchmark to compare the times for computing expressions by using
# ctable objects.  Numexpr is needed in order to execute this.

import math
from time import time

import numpy as np
import numexpr as ne

import bcolz


N = 1e7  # the number of elements in x
clevel = 3  # the compression level
# sexpr = "(x+1)<0"  # the expression to compute
#sexpr = "(2*x**3+.3*y**2+z+1)<0"  # the expression to compute
sexpr = "((.25*x + .75)*x - 1.5)*x - 2"  # a computer-friendly polynomial
#sexpr = "(((.25*x + .75)*x - 1.5)*x - 2)<0"  # a computer-friendly polynomial

print("Creating inputs...")

cparams = bcolz.cparams(clevel)

x = np.arange(N)
#x = np.linspace(0,100,N)
cx = bcolz.carray(x, cparams=cparams)
if 'y' not in sexpr:
    t = bcolz.ctable((cx,), names=['x'])
else:
    y = np.arange(N)
    z = np.arange(N)
    cy = bcolz.carray(y, cparams=cparams)
    cz = bcolz.carray(z, cparams=cparams)
    t = bcolz.ctable((cx, cy, cz), names=['x', 'y', 'z'])

print("Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N))))

t0 = time()
out = eval(sexpr)
print("Time for plain numpy--> %.3f" % (time() - t0,))

t0 = time()
out = ne.evaluate(sexpr)
print("Time for numexpr (numpy)--> %.3f" % (time() - t0,))

# Uncomment the next for disabling threading
#ne.set_num_threads(1)
#bcolz.blosc_set_nthreads(1)
# Seems that this works better if we dividw the number of cores by 2.
# Maybe due to some contention between Numexpr and Blosc?
#bcolz.set_nthreads(bcolz.ncores//2)

for kernel in "python", "numexpr":
    t0 = time()
    #cout = t.eval(sexpr, kernel=kernel, cparams=cparams)
    cout = t.eval(sexpr, cparams=cparams)
    print("Time for ctable (%s) --> %.3f" % (kernel, time() - t0,))
    #print "cout-->", repr(cout)

#assert_array_equal(out, cout, "Arrays are not equal")
