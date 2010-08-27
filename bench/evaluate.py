# Benchmark to compare the times for computing expressions by using
# carrays vs plain numpy arrays.  The tables.Expr class is used for
# this.

import math
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import tables as tb
import carray as ca
from time import time

N = 1e7       # the number of elements in x
clevel = 5    # the compression level
#sexpr = "x+1"  # the expression to compute
sexpr = "2*x**3+.3*x**2+x+1"  # the expression to compute

print "Evaluating '%s' with 10**%d points" % (sexpr, int(math.log10(N)))

# Create the numpy array
x = np.arange(N)
# Create a compressed array
cx = ca.carray(x, clevel=clevel)
cout = ca.carray(np.empty((0,), dtype='f8'), clevel=clevel)

t0 = time()
out = eval(sexpr)
print "Time for plain numpy--> %.3f" % (time()-t0,) 

t0 = time()
out = ne.evaluate(sexpr)
print "Time for numexpr (numpy)--> %.3f" % (time()-t0,) 

t0 = time()
expr = tb.Expr(sexpr)
out = expr.eval()
print "Time for tables.Expr (numpy)--> %.3f" % (time()-t0,) 

x = cx
t0 = time()
expr = tb.Expr(sexpr)
expr.setOutput(cout, append_mode=True)
expr.eval()
print "Time for compressed array--> %.3f" % (time()-t0,) 
print "cout-->", repr(cout)

#assert_array_equal(out, cout.toarray(), "Arrays are not equal")
