from __future__ import print_function

import itertools
import numpy as np
import numexpr as ne
import bcolz
import time
import cProfile
import inspect

print("numexpr version:", ne.__version__)
bcolz.defaults.cparams['shuffle'] = bcolz.SHUFFLE
#bcolz.defaults.cparams['shuffle'] = bcolz.BITSHUFFLE
bcolz.defaults.cparams['cname'] = 'blosclz'
#bcolz.defaults.cparams['cname'] = 'lz4'
bcolz.defaults.cparams['clevel'] = 5
#bcolz.defaults.vm = "dask"
#bcolz.defaults.vm = "python"
bcolz.defaults.vm = "numexpr"

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
        print(f.__name__, 'took', round(end - start, 3), 'sec')
        return result
    return f_timer


@timefunc
def where_numpy():
    return sum(npa[i] for i in np.where((npa > 5) & (npb < LMAX))[0])

@timefunc
def where_numexpr():
    return sum(npa[i] for i in np.where(
        ne.evaluate('(npa > 5) & (npb < LMAX)'))[0])

@timefunc
#@do_cprofile
def bcolz_where():
    return sum(r.a for r in ct.where("(a > 5) & (b < LMAX)"))

@timefunc
#@do_cprofile
def bcolz_where_numpy():
    return sum(r.a for r in ct.where("(npa > 5) & (npb < LMAX)"))

@timefunc
#@do_cprofile
def bcolz_where_numexpr():
    return sum(r.a for r in ct.where(ne.evaluate("(npa > 5) & (npb < LMAX)")))

@timefunc
#@do_cprofile
def whereblocks():
    sum = 0.
    for r in ct.whereblocks("(a > 5) & (b < LMAX)", blen=None):
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

@timefunc
#@do_cprofile
def fetchwhere_dask():
    result = ct.fetchwhere("(a > 5) & (b < LMAX)", vm="dask")['a'].sum()
    return result


print(repr(ct))

a0 = where_numpy()
print("a0:", a0)
a1 = where_numexpr()
assert a0 == a1
a1 = bcolz_where()
assert a0 == a1
a1 = bcolz_where_numpy()
assert a0 == a1
a1 = bcolz_where_numexpr()
assert a0 == a1
a1 = whereblocks()
assert a0 == a1
a1 = fetchwhere_bcolz()
assert a0 == a1
a1 = fetchwhere_numpy()
assert a0 == a1
a1 = fetchwhere_dask()
assert a0 == a1
