import numpy as np
import carray as ca
from time import time

N = 1 * 1000 * 1000

a = np.linspace(0, 1, N)

t0 = time()
ac = ca.carray(a, cparams=ca.cparams(clevel=5))
print "time creation (memory) ->", round(time()-t0, 3)
print "data (memory):", repr(ac)

t0 = time()
b = ca.carray(a, cparams=ca.cparams(clevel=5), rootdir='myarray')
b.flush()
print "time creation (disk) ->", round(time()-t0, 3)
#print "meta (disk):", b.read_meta()
print "data (disk):", repr(b)

t0 = time()
an = np.array(a)
print "time creation (numpy) ->", round(time()-t0, 3)

t0 = time()
c = ca.carray(rootdir='myarray')
print "time open (disk) ->", round(time()-t0, 3)
#print "meta (disk):", c.read_meta()
#print "data (disk):", sum(c)

t0 = time()
sum(ac)
print "time sum (memory) ->", round(time()-t0, 3)

t0 = time()
sum(c)
print "time sum (disk) ->", round(time()-t0, 3)

t0 = time()
sum(a)
print "time sum (numpy) ->", round(time()-t0, 3)
