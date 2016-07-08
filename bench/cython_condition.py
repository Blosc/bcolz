import numpy as np
import pyorcy
#c import cython


@pyorcy.cythonize #p
def condition(a, b): #p
#c @cython.boundscheck(False)
#c def condition(double[:] a, double[:] b):
    #c cdef bint [:] res = np.empty(len(a), dtype=np.int32)
    #c for i in range(len(a)):
    #c    res[i] = (a[i] > 5.) & (b[i] < 1e3)
    #c return np.asarray(res, dtype=np.bool)
    return (a > 5.) & (b < 1e3)
