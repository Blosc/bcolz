import numpy as np
import carray as ca
from time import time

N = 1e8
a = np.arange(N, dtype='i4')

t0 = time()
sa = a.sum()
print "Time sum() numpy --> %.3f" % (time()-t0)

ac = ca.carray(a, cparams=ca.cparams(9))
print "ac-->", `ac`

t0 = time()
sac = ac.sum()
print "Time sum() carray --> %.3f" % (time()-t0)

# t0 = time()
# sac = sum(i for i in ac)
# print "Time sum() carray (iter) --> %.3f" % (time()-t0)

assert(sa == sac)
