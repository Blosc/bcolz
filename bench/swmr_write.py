# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from time import time
import threading


import bcolz
from bcolz.swmr import SWMRWrapper
import numpy as np


class PartialCopy(threading.Thread):

    def __init__(self, x, y, chunk_start=0, chunk_step=1):
        super(PartialCopy, self).__init__()
        self.x = x
        self.y = y
        self.chunk_start = chunk_start
        self.chunk_step = chunk_step
        self.result = 0

    def run(self):
        x = self.x
        y = self.y
        chunklen = x.chunklen
        n_chunks = int(np.ceil(x.shape[0] / chunklen))
        for i in range(self.chunk_start, n_chunks, self.chunk_step):
            start = i * chunklen
            stop = (i * chunklen) + chunklen
            y[start:stop] = x[start:stop]


def parallel_copy(x, y, num_workers=1):
    workers = [PartialCopy(x, y, chunk_start=i, chunk_step=num_workers)
               for i in range(num_workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


def run_benchmark(bcolz_nthreads=None):
    if bcolz_nthreads is None:
        bcolz_nthreads = bcolz.detect_number_of_cores()
    print('Bcolz threads: %s' % bcolz_nthreads)
    bcolz.set_nthreads(bcolz_nthreads)

    x = bcolz.arange(0, 1e9, 1, dtype='i4')
    y = bcolz.zeros(x.shape, dtype=x.dtype)
    xs = SWMRWrapper(x)
    ys = SWMRWrapper(y)

    w = PartialCopy(x, y)
    t0 = time()
    w.run()
    assert x[0] == y[0]
    assert x[-1] == y[-1]
    print("Time non-threaded --> %.3f" % (time() - t0))

    for n in range(1, bcolz.detect_number_of_cores() + 1):
        t0 = time()
        parallel_copy(xs, ys, num_workers=n)
        assert x[0] == y[0]
        assert x[-1] == y[-1]
        t1 = time()
        print("Time parallel_copy(num_workers=%s) --> %.3f" % (n, (t1 - t0)))


run_benchmark(1)
run_benchmark()
