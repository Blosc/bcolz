import numpy as np
import carray as ca
from time import time

N = 1e8
dtype = 'i4'

t0 = time()
a = np.zeros(N, dtype=dtype)
print "Time numpy.zeros() --> %.4f" % (time()-t0)

t0 = time()
ac = ca.zeros(N, dtype=dtype)
#ac = ca.carray(a)
print "Time carray.zeros() --> %.4f" % (time()-t0)

print "ac-->", `ac`

#assert(np.all(a == ac))
