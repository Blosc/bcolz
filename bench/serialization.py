import numpy as np
import carray as ca
from time import time

N = 1000 * 1000

a = np.linspace(0, 1, N)
ac = ca.carray(a, cparams=ca.cparams(clevel=5))

b = ca.carray(a, cparams=ca.cparams(clevel=5), rootdir='proves')
b.flush()
print "meta (memory):", b.read_meta()
print "data (memory):", sum(b)

c = ca.carray(rootdir='proves')
print "meta (disk):", c.read_meta()
print "data (disk):", sum(c)

t0 = time()
sum(ac)
print "t (memory) ->", round(time()-t0, 3)
t0 = time()
sum(c)
print "t (disk) ->", round(time()-t0, 3)

t0 = time()
sum(a)
print "t (numpy) ->", round(time()-t0, 3)
