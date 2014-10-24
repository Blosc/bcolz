# Benchmark for pickling carray/ctable objects

from __future__ import print_function

import os.path
import shutil
import contextlib
import time
import pickle
import numpy as np
import numpy.testing

import bcolz


@contextlib.contextmanager
def ctime(message=None):
    "Counts the time spent in some context"
    t = time.time()
    yield
    if message:
        print(message + ":\t", end="")
    print(round(time.time() - t, 4), "sec")


carootdir = "carraypickle.bcolz"
if os.path.exists(carootdir):
    shutil.rmtree(carootdir)
ctrootdir = "ctablepickle.bcolz"
if os.path.exists(ctrootdir):
    shutil.rmtree(ctrootdir)

N = int(1e7)
a = bcolz.arange(N, dtype="int32")
b = bcolz.arange(N, dtype="float32")
ca = bcolz.carray(a, rootdir=carootdir)
ct = bcolz.ctable([ca, b], names=['a', 'b'], rootdir=ctrootdir)

with ctime("Time spent pickling carray with N=%d" % N):
    s = pickle.dumps(ca)

with ctime("Time spent unpickling carray with N=%d" % N):
    ca2 = pickle.loads(s)

np.testing.assert_allclose(ca2[:], a)

with ctime("Time spent pickling ctable with N=%d" % N):
    s = pickle.dumps(ct)

with ctime("Time spent unpickling ctable with N=%d" % N):
    ct2 = pickle.loads(s)

np.testing.assert_allclose(ct2['a'][:], a)
np.testing.assert_allclose(ct2['b'][:], b)
