from __future__ import print_function
import contextlib
import time


@contextlib.contextmanager
def ctime(label=""):
    "Counts the time spent in some context"
    t = time.time()
    yield
    print(label, round(time.time() - t, 3), "sec")
