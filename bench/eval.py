# Benchmark to compare the times for computing expressions by using
# eval() on carray/numpy arrays.  Numexpr is needed in order to
# execute this.

import math
import numpy as np
import numexpr as ne
import carray as ca
from time import time

N = 1e7       # the number of elements in x
clevel = 9    # the compression level
sexprs = [ "(x+1)<0",
           "(2*x**2+.3*y**2+z+1)<0",
           "((.25*x + .75)*x - 1.5)*x - 2",
           "(((.25*x + .75)*x - 1.5)*x - 2)<0",
           ]

# Initial dataset
#x = np.arange(N)
x = np.linspace(0,100,N)

doprofile = True

def compute_ref(sexpr):
    t0 = time()
    out = eval(sexpr)
    print "Time for plain numpy --> %.3f" % (time()-t0,)

    t0 = time()
    out = ne.evaluate(sexpr)
    print "Time for numexpr (numpy) --> %.3f" % (time()-t0,)

def compute_carray(sexpr, clevel, kernel):
    # Uncomment the next for disabling threading
    # Maybe due to some contention between Numexpr and Blosc?
    # ca.set_nthreads(ca.ncores//2)
    print "*** carray (using compression clevel = %d):" % clevel
    if clevel > 0:
        x, y, z = cx, cy, cz
    t0 = time()
    cout = ca.eval(sexpr, kernel=kernel, cparams=ca.cparams(clevel))
    print "Time for ca.eval (%s) --> %.3f" % (kernel, time()-t0,),
    print ", cratio (out): %.1f" % (cout.nbytes / float(cout.cbytes))
    #print "cout-->", repr(cout)


if __name__=="__main__":

    print "Creating inputs..."

    cparams = ca.cparams(clevel)

    y = x.copy()
    z = x.copy()
    cx = ca.carray(x, cparams=cparams)
    cy = ca.carray(y, cparams=cparams)
    cz = ca.carray(z, cparams=cparams)

    for sexpr in sexprs:
        print "Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N)))
        compute_ref(sexpr)
        for kernel in "python", "numexpr":
            compute_carray(sexpr, clevel=0, kernel=kernel)
        if doprofile:
            import pstats
            import cProfile as prof
            #prof.run('compute_carray(sexpr, clevel=clevel, kernel="numexpr")',
            prof.run('compute_carray(sexpr, clevel=clevel, kernel="python")',
                     'eval.prof')
            stats = pstats.Stats('eval.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            for kernel in "python", "numexpr":
                compute_carray(sexpr, clevel=clevel, kernel=kernel)
