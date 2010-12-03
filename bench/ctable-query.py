# Benchmark to compare the times for computing expressions by using
# ctable objects.  Numexpr is needed in order to execute this.

import math
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import carray as ca
from time import time

#NR = int(1e5) # the number of rows
NR = int(1e5) # the number of rows
NC = 500      # the number of columns
clevel = 9    # the compression level
squery = "((((f0+f1)<3) & ((f30+f31)<3e3)) | ((f2>3) & (f2<3e4)))"  # the query
nquery = "((((t['f0']+t['f1'])<3) & ((t['f30']+t['f31'])<3e3)) | ((t['f2']>3) & (t['f2']<3e4)))"  # the query for a recarray
#squery = "((f2>3) & (f2<3e4))"  # the query
#nquery = "((t['f2']>3) & (t['f2']<3e4))"  # the query for a recarray

print "Creating inputs..."

cparams = ca.cparams(clevel)

x = np.arange(NR)
#x = np.linspace(0,100,NR)
# Create a recarray made by copies of x
t = np.rec.fromarrays([x]*NC)
# Create a ctable out of the recarray
tc = ca.ctable(t)

print "Evaluating '%s' with 10^%d rows" % (squery, int(math.log10(NR)))

t0 = time()
out = t[eval(nquery)][['f0','f2']]
print "Time for plain numpy --> %.3f" % (time()-t0,)

map_field = dict(("f%s"%i, t["f%s"%i]) for i in range(NC))
t0 = time()
out = t[ne.evaluate(squery, map_field)][['f0','f2']]
print "Time for numexpr --> %.3f" % (time()-t0,)

# Uncomment the next for disabling threading
#ne.set_nthreads(1)
#ca.blosc_set_num_threads(1)
# Seems that this works better if we dividw the number of cores by 2.
# Maybe due to some contention between Numexpr and Blosc?
ca.set_nthreads(ca.ncores//2)

t0 = time()
#cout = tc.eval(sexpr, cparams=cparams)
cout = [row for row in tc.getif(squery, ['f0','f2'])]
print "Time for ctable--> %.3f" % (time()-t0,)
#print "cout-->", repr(cout)

#assert_array_equal(out, cout, "Arrays are not equal")
