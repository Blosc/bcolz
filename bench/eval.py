# Benchmark to compare the times for computing expressions by using
# eval() on carray/numpy arrays.  Numexpr is needed in order to
# execute this.

from __future__ import print_function

import math
from time import time

import numpy as np
import numexpr as ne

import bcolz


N = 1e7  # the number of elements in x
clevel = 9  # the compression level
sexprs = ["(x+1)<0",
          "(2*x**2+.3*y**2+z+1)<0",
          "((.25*x + .75)*x - 1.5)*x - 2",
          "(((.25*x + .75)*x - 1.5)*x - 2)<0",
]

# Initial dataset
# x = np.arange(N)
x = np.linspace(0, 100, N)

doprofile = False


def compute_ref(sexpr):
    t0 = time()
    out = eval(sexpr)
    print("Time for plain numpy --> %.3f" % (time() - t0,))

    t0 = time()
    out = ne.evaluate(sexpr)
    print("Time for numexpr (numpy) --> %.3f" % (time() - t0,))


def compute_carray(sexpr, clevel, vm):
    # Uncomment the next for disabling threading
    # Maybe due to some contention between Numexpr and Blosc?
    # bcolz.set_nthreads(bcolz.ncores//2)
    print("*** carray (using compression clevel = %d):" % clevel)
    if clevel > 0:
        x, y, z = cx, cy, cz
    t0 = time()
    cout = bcolz.eval(sexpr, vm=vm, cparams=bcolz.cparams(clevel))
    print("Time for bcolz.eval (%s) --> %.3f" % (vm, time() - t0,), end="")
    print(", cratio (out): %.1f" % (cout.nbytes / float(cout.cbytes)))
    #print "cout-->", repr(cout)


if __name__ == "__main__":

    print("Creating inputs...")

    cparams = bcolz.cparams(clevel)

    y = x.copy()
    z = x.copy()
    cx = bcolz.carray(x, cparams=cparams)
    cy = bcolz.carray(y, cparams=cparams)
    cz = bcolz.carray(z, cparams=cparams)

    for sexpr in sexprs:
        print("Evaluating '%s' with 10^%d points" % (
            sexpr, int(math.log10(N))))
        compute_ref(sexpr)
        for vm in "python", "numexpr":
            compute_carray(sexpr, clevel=0, vm=vm)
        if doprofile:
            import pstats
            import cProfile as prof
            #prof.run('compute_carray(sexpr, clevel=clevel, vm="numexpr")',
            prof.run('compute_carray(sexpr, clevel=0, vm="numexpr")',
                     #prof.run('compute_carray(sexpr, clevel=clevel,
                     # vm="python")',
                     #prof.run('compute_carray(sexpr, clevel=0, vm="python")',
                     'eval.prof')
            stats = pstats.Stats('eval.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            stats.print_stats(20)
        else:
            for vm in "python", "numexpr":
                compute_carray(sexpr, clevel=clevel, vm=vm)
