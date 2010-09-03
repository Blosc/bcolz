# Benchmark to compare the times for computing expressions by using
# ctable objects.  Numexpr is needed in order to execute this.

import math
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import carray as ca
from time import time

# Uncomment the next for disabling threading
#ne.set_num_threads(1)
#ca.setBloscMaxThreads(1)

N = 1e7       # the number of elements in x
clevel = 5    # the compression level
#sexpr = "x+1"  # the expression to compute
sexpr = "2*x**3+.3*y**2+z+1"  # the expression to compute

print "Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N)))

# Create the numpy arrays
x = np.arange(N)
y = np.arange(N)
z = np.arange(N)

t0 = time()
out = eval(sexpr)
print "Time for plain numpy--> %.3f" % (time()-t0,)

t0 = time()
out = ne.evaluate(sexpr)
print "Time for numexpr (numpy)--> %.3f" % (time()-t0,)

t0 = time()
if sexpr == "x+1":
    t = ca.ctable((x,), names=['x'])
else:
    t = ca.ctable((x,y,z), names=['x','y','z'])
cout = t.eval(sexpr)
print "Time for ctable--> %.3f" % (time()-t0,)
print "cout-->", repr(cout)

#assert_array_equal(out, cout, "Arrays are not equal")
