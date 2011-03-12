# Benchmark to compare the times for computing expressions by using
# eval() on carray/numpy arrays.  Numexpr is needed in order to
# execute this.

import math
import numpy as np
import numexpr as ne
import carray as ca
from time import time

N = 1e4       # the number of rows in x
clevel = 9    # the compression level

# Initial dataset
x = np.eye(N)
sexprs = ["x*x"]

doprofile = False

def compute_ref(sexpr):
    t0 = time()
    out = eval(sexpr)
    print "Time for plain numpy --> %.3f" % (time()-t0,)

    t0 = time()
    out = ne.evaluate(sexpr)
    print "Time for numexpr (numpy) --> %.3f" % (time()-t0,)

def compute_carray(sexpr, clevel, vm):
    # Uncomment the next for disabling threading
    # Maybe due to some contention between Numexpr and Blosc?
    # ca.set_nthreads(ca.ncores//2)
    print "*** carray (using compression clevel = %d):" % clevel
    if clevel > 0:
        x = cx
    t0 = time()
    cout = ca.eval(sexpr, vm=vm, cparams=ca.cparams(clevel))
    print "Time for ca.eval (%s) --> %.3f" % (vm, time()-t0,),
    print ", cratio (out): %.1f" % (cout.nbytes / float(cout.cbytes))
    #print "cout-->", repr(cout)


if __name__=="__main__":

    cx = ca.carray(x, cparams=ca.cparams(clevel=clevel))

    for sexpr in sexprs:
        xpo = int(math.log10(N))
        print "Evaluating '%s' with 10^%d x 10^%d" % (sexpr, xpo, xpo)
        compute_ref(sexpr)
        if doprofile:
            import pstats
            import cProfile as prof
            #prof.run('compute_carray(sexpr, clevel=clevel, vm="numexpr")',
            prof.run('compute_carray(sexpr, clevel=0, vm="numexpr")',
            #prof.run('compute_carray(sexpr, clevel=clevel, vm="python")',
            #prof.run('compute_carray(sexpr, clevel=0, vm="python")',
                     'eval.prof')
            stats = pstats.Stats('eval.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            for vm in "python", "numexpr":
                compute_carray(sexpr, clevel=clevel, vm=vm)
