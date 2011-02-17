import numpy as np
import carray as ca
from time import time

N = 1e7
T = 100
clevel=0
np.random.seed(1)
a = np.random.normal(scale=100,size=N)

t0 = time()
sa = np.where(a>T)[0][-1]
print "Time where numpy --> %.3f" % (time()-t0)

t0 = time()
iac = ca.arange(int(N), cparams=ca.cparams(clevel))
ac = ca.eval("a>T", cparams=ca.cparams(clevel))
sac1 = [r for r in iac.where(ac)][-1]
print "Time where carray --> %.3f" % (time()-t0)

t0 = time()
ac = ca.eval("a>T", cparams=ca.cparams(clevel))
sac = [r for r in ac.wheretrue()][-1]
print "Time wheretrue carray --> %.3f" % (time()-t0)

t0 = time()
iac = ca.arange(int(N), cparams=ca.cparams(clevel))
ac = ca.eval("a>T", cparams=ca.cparams(clevel))
sac3 = [r for r in iac.where(ac, skip=-1)][0]
print "Time where carray, skip --> %.3f" % (time()-t0)

t0 = time()
ac = ca.eval("a>T", cparams=ca.cparams(clevel))
sac2 = [r for r in ac.wheretrue(skip=-1)][0]
print "Time wheretrue carray, skip --> %.3f" % (time()-t0)

print "sa, sac, sac1, sca2, sac3-->", sa, sac1, sac, sac2, sac3
assert(sa == sac)
assert(sa == sac1)
assert(sa == sac2)
assert(sa == sac3)
