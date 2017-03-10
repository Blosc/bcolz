# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from time import time
import threading


import bcolz
from bcolz.swmr import SWMRWrapper
import numpy as np


class PartialSum(threading.Thread):

    def __init__(self, x, chunk_start=0, chunk_step=1):
        super(PartialSum, self).__init__()
        self.x = x
        self.chunk_start = chunk_start
        self.chunk_step = chunk_step
        self.result = 0

    def run(self):
        x = self.x
        chunklen = x.chunklen
        n_chunks = int(np.ceil(x.shape[0] / chunklen))
        for i in range(self.chunk_start, n_chunks, self.chunk_step):
            start = i * chunklen
            stop = (i * chunklen) + chunklen
            a = x[start:stop]
            self.result += a.sum()


def parallel_sum(x, num_workers=1):
    workers = [PartialSum(x, chunk_start=i, chunk_step=num_workers)
               for i in range(num_workers)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return sum(w.result for w in workers)


def run_benchmark(bcolz_nthreads=None):
    if bcolz_nthreads is None:
        bcolz_nthreads = bcolz.detect_number_of_cores()
    print('Bcolz threads: %s' % bcolz_nthreads)
    bcolz.set_nthreads(bcolz_nthreads)

    c = bcolz.arange(0, 1e9, 1, dtype='i4')
    cs = SWMRWrapper(c)

    t0 = time()
    s = c.sum()
    print("Time carray.sum() --> %.3f" % (time() - t0))

    t0 = time()
    w = PartialSum(c)
    w.run()
    ps = w.result
    assert s == ps
    print("Time non-threaded sum() --> %.3f" % (time() - t0))

    for n in range(1, bcolz.detect_number_of_cores() + 1):
        t0 = time()
        ps = parallel_sum(cs, num_workers=n)
        assert s == ps
        t1 = time()
        print("Time parallel_sum(num_workers=%s) --> %.3f" % (n, (t1 - t0)))


run_benchmark(1)
run_benchmark()
