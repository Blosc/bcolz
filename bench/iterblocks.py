import numpy as np
import bcolz
import time


bcolz.defaults.cparams['shuffle'] = bcolz.SHUFFLE
# bcolz.defaults.cparams['shuffle'] = bcolz.BITSHUFFLE
bcolz.defaults.cparams['cname'] = 'blosclz'
# bcolz.defaults.cparams['cname'] = 'lz4'
bcolz.defaults.cparams['clevel'] = 9

N = int(1e8)
a = np.arange(N)
ca = bcolz.carray(a)


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', round(end - start, 3), 'sec')
        return result
    return f_timer


@timefunc
def iterblocks0():
    return a.sum()


@timefunc
def iterblocks1(arr):
    return sum(i for i in arr)


@timefunc
def iterblocks2(arr):
    sum_ = 0.
    for b in bcolz.iterblocks(arr, blen=arr.chunklen):
        sum_ += b.sum()
    return sum_


print(repr(ca))

a0 = iterblocks0()
print("a0:", a0)
# a1 = iterblocks1(ca)
# assert a0 == a1
a1 = iterblocks2(ca)
assert a0 == a1
