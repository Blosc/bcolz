from contextlib import contextmanager
import numpy as np
import time
from bcolz import carray
from bcolz.carray_ext import test_v1, test_v2, test_v3

@contextmanager
def ctime(message=None):
    assert message is not None
    "Counts the time spent in some context"
    t = time.time()
    yield
    print message + ":\t", \
          round(time.time() - t, 4), "sec"

n = 50033

x = np.arange(0, n, 1)
c = carray(x, dtype='int64', chunklen=100)

with ctime('test_v1'):
    r1 = test_v1(c)

with ctime('test_v2'):
    r2 = test_v2(c)

with ctime('test_v3'):
    r3 = test_v3(c)

assert np.array_equal(r1, r2)
assert np.array_equal(r1, r3)
