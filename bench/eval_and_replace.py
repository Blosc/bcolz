from time import time
import numpy as np
import bcolz

N = int(1e6)

# Initial dataset
x = np.linspace(0, 100, N)

cparams = bcolz.cparams(clevel=5)

cx = bcolz.carray(x, cparams=cparams)
cy = bcolz.carray(x+1, cparams=cparams)
cz = bcolz.carray(x+2, cparams=cparams)

ct = bcolz.ctable([cx, cy, cz])
t0 = time()
#ct['f0'] = ct.eval('f0 + 1', cparams=cparams)
ct['f0'] = x + 1
print("Time for computation --> %.3f" % (time() - t0,))
print(repr(ct['f0']))

#np.testing.assert_allclose(ct['f0'], ct['f1'])
