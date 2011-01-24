# Benchmark to compare the times for computing expressions by using
# eval() on carray/numpy arrays.  Numexpr is needed in order to
# execute this.

import math
import numpy as np
import numexpr as ne
import carray as ca
from time import time

def compute_carray(sexpr, clevel, kernel):
    # Uncomment the next for disabling threading
    ca.set_nthreads(1)
    ca.blosc_set_nthreads(1)
    print("*** carray (using compression clevel = %d):" % clevel)
    x = cx
    t0 = time()
    cout = ca.eval(sexpr, kernel=kernel, cparams=ca.cparams(clevel))
    print("Time for ca.eval (%s) --> %.3f" % (kernel, time()-t0,))
    #print(", cratio (out): %.1f" % (cout.nbytes / float(cout.cbytes)))
    #print "cout-->", repr(cout)


if __name__=="__main__":

    N = 1e7       # the number of elements in x
    clevel = 0    # the compression level
    sexpr = "(x+1)<0"
    sexpr = "(((.25*x + .75)*x - 1.5)*x - 2)<0"
    doprofile = 1

    print("Creating inputs...")
    #x = np.arange(N)
    x = np.linspace(0,100,N)
    cx = ca.carray(x, cparams=ca.cparams(clevel))

    print("Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N))))

    # t0 = time()
    # cout = ne.evaluate(sexpr)
    # print "Time for numexpr --> %.3f" % (time()-t0,)

    if doprofile:
        import pstats
        import cProfile as prof
        prof.run('compute_carray(sexpr, clevel=clevel, kernel="numexpr")',
        #prof.run('compute_carray(sexpr, clevel=clevel, kernel="python")',
                 'eval.prof')
        stats = pstats.Stats('eval.prof')
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(20)
    else:
        compute_carray(sexpr, clevel=clevel, kernel="numexpr")
        #compute_carray(sexpr, clevel=clevel, kernel="python")
