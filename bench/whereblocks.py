import itertools
import numpy as np
import numexpr as ne
import bcolz
import time
import cProfile
import inspect


bcolz.defaults.cparams['shuffle'] = bcolz.SHUFFLE
#bcolz.defaults.cparams['shuffle'] = bcolz.BITSHUFFLE
#bcolz.defaults.cparams['cname'] = 'blosclz'
bcolz.defaults.cparams['cname'] = 'lz4'
bcolz.defaults.cparams['clevel'] = 5

N = 1e8
LMAX = 1e3
npa = np.arange(N)
npb = np.arange(N)
ct = bcolz.ctable([npa, npb], names=["a", "b"])


def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='cumulative')
    return profiled_func



def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', round(end - start, 3), 'sec'
        return result
    return f_timer


@timefunc
def where0():
    return sum(npa[i] for i in np.where((npa > 5) & (npb < LMAX))[0])

@timefunc
#@do_cprofile
def where1():
    return sum(r[0] for r in ct.where("(npa > 5) & (npb < LMAX)",
                                      out_flavor=tuple))
@timefunc
#@do_cprofile
def where2():
    return sum(r[0] for r in ct.where("(a > 5) & (b < LMAX)",
                                      out_flavor=tuple))

@timefunc
#@do_cprofile
def whereblocks():
    sum = 0.
    for r in ct.whereblocks("(a > 5) & (b < LMAX)", blen=None):
    #for r in ct.whereblocks("(a > 5) & (b < LMAX)", blen=ct['a'].chunklen*10):
    #for r in ct.whereblocks("(a > 5) & (b < LMAX)", blen=1000):
        sum += r['a'].sum()
    return sum

@timefunc
#@do_cprofile
def fetchwhere_bcolz():
    return ct.fetchwhere("(a > 5) & (b < LMAX)", out_flavor='bcolz')['a'].sum()

@timefunc
#@do_cprofile
def fetchwhere_numpy():
    return ct.fetchwhere("(a > 5) & (b < LMAX)", out_flavor='numpy')['a'].sum()


print repr(ct)

a0 = where0()
print "a0:", a0
a1 = where1()
assert a0 == a1
a1 = where2()
assert a0 == a1
a1 = whereblocks()
assert a0 == a1
a1 = fetchwhere_bcolz()
assert a0 == a1
a1 = fetchwhere_numpy()
assert a0 == a1
