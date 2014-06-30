from time import time

import numpy as np

import bcolz


N = int(1e7)
CLEVEL = 5

a = np.linspace(0, 1, N)

t0 = time()
ac = bcolz.carray(a, cparams=bcolz.cparams(clevel=CLEVEL))
print("time creation (memory) ->", round(time() - t0, 3))
print("data (memory):", repr(ac))

t0 = time()
b = bcolz.carray(a, cparams=bcolz.cparams(clevel=CLEVEL),
                 rootdir='myarray', mode='w')
b.flush()
print("time creation (disk) ->", round(time() - t0, 3))
# print "meta (disk):", b.read_meta()

t0 = time()
an = np.array(a)
print("time creation (numpy) ->", round(time() - t0, 3))

t0 = time()
c = bcolz.carray(rootdir='myarray')
print("time open (disk) ->", round(time() - t0, 3))
#print "meta (disk):", c.read_meta()
print("data (disk):", repr(c))

t0 = time()
print(sum(ac))
print("time sum (memory, iter) ->", round(time() - t0, 3))

t0 = time()
print(sum(c))
print("time sum (disk, iter) ->", round(time() - t0, 3))

t0 = time()
print(bcolz.eval('sum(ac)'))
print("time sum (memory, eval) ->", round(time() - t0, 3))

t0 = time()
print(bcolz.eval('sum(c)'))
print("time sum (disk, eval) ->", round(time() - t0, 3))

t0 = time()
print(ac.sum())
print("time sum (memory, method) ->", round(time() - t0, 3))

t0 = time()
print(c.sum())
print("time sum (disk, method) ->", round(time() - t0, 3))

t0 = time()
print(a.sum())
print("time sum (numpy, method) ->", round(time() - t0, 3))
