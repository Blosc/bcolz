# Benchmark to compare the times for computing expressions by using
# eval() on bcolz/numpy arrays.  Numexpr is needed in order to
# execute this.

import math
from time import time

import numpy as np
import numexpr as ne

import bcolz


def compute_bcolz(sexpr, clevel, vm):
    # Uncomment the next for disabling threading
    # bcolz.set_nthreads(1)
    #bcolz.blosc_set_nthreads(1)
    print("*** bcolz (using compression clevel = %d):" % clevel)
    x = cx  # comment this for using numpy arrays in inputs
    t0 = time()
    cout = bcolz.eval(sexpr, vm=vm, cparams=bcolz.cparams(clevel))
    print("Time for bcolz.eval (%s) --> %.3f" % (vm, time() - t0,))
    #print(", cratio (out): %.1f" % (cout.nbytes / float(cout.cbytes)))
    #print("cout-->", repr(cout))


if __name__ == "__main__":

    N = 1e8  # the number of elements in x
    clevel = 3  # the compression level
    sexpr = "(x+1)<0"
    sexpr = "(((.25*x + .75)*x - 1.5)*x - 2)<0"
    # sexpr = "(((.25*x + .75)*x - 1.5)*x - 2)"
    doprofile = 0

    print("Creating inputs...")
    x = np.arange(N)
    #x = np.linspace(0,100,N)
    cx = bcolz.carray(x, cparams=bcolz.cparams(clevel))

    print("Evaluating '%s' with 10^%d points" % (sexpr, int(math.log10(N))))

    t0 = time()
    cout = ne.evaluate(sexpr)
    print("Time for numexpr --> %.3f" % (time() - t0,))

    if doprofile:
        import pstats
        import cProfile as prof

        prof.run('compute_bcolz(sexpr, clevel=clevel, vm="numexpr")',
                 #prof.run('compute_bcolz(sexpr, clevel=clevel, vm="python")',
                 'eval.prof')
        stats = pstats.Stats('eval.prof')
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        stats.print_stats(20)
    else:
        compute_bcolz(sexpr, clevel=clevel, vm="numexpr")
        #compute_bcolz(sexpr, clevel=clevel, vm="python")
