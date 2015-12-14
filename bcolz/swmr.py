# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import contextlib
import threading


class SWMRLock(object):
    """
    Classic implementation of reader-writer lock with preference to writers.
    Readers can access a resource simultaneously. Writers get an exclusive
    access.

    Copied from django.utils.synch module, contributed to django by
    eugene@lazutkin.com.

    """

    def __init__(self):
        self.mutex = threading.RLock()
        self.can_read = threading.Semaphore(0)
        self.can_write = threading.Semaphore(0)
        self.active_readers = 0
        self.active_writers = 0
        self.waiting_readers = 0
        self.waiting_writers = 0

    def reader_enters(self):
        with self.mutex:
            if self.active_writers == 0 and self.waiting_writers == 0:
                self.active_readers += 1
                self.can_read.release()
            else:
                self.waiting_readers += 1
        self.can_read.acquire()

    def reader_leaves(self):
        with self.mutex:
            self.active_readers -= 1
            if self.active_readers == 0 and self.waiting_writers != 0:
                self.active_writers += 1
                self.waiting_writers -= 1
                self.can_write.release()

    @contextlib.contextmanager
    def reader(self):
        self.reader_enters()
        try:
            yield
        finally:
            self.reader_leaves()

    def writer_enters(self):
        with self.mutex:
            if self.active_writers == 0 and self.waiting_writers == 0 and self.active_readers == 0:
                self.active_writers += 1
                self.can_write.release()
            else:
                self.waiting_writers += 1
        self.can_write.acquire()

    def writer_leaves(self):
        with self.mutex:
            self.active_writers -= 1
            if self.waiting_writers != 0:
                self.active_writers += 1
                self.waiting_writers -= 1
                self.can_write.release()
            elif self.waiting_readers != 0:
                t = self.waiting_readers
                self.waiting_readers = 0
                self.active_readers += t
                while t > 0:
                    self.can_read.release()
                    t -= 1

    @contextlib.contextmanager
    def writer(self):
        self.writer_enters()
        try:
            yield
        finally:
            self.writer_leaves()


class SWMRWrapper(object):
    """Wrapper for a bcolz carray or ctable using a reader-writer lock to
    manage data access and update operations.

    Parameters
    ----------
    arr : array_like
        Array-like containing underlying data, e.g., a bcolz carray or
        ctable.

    """

    def __init__(self, arr):
        self.arr = arr
        self.lock = SWMRLock()

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    @property
    def chunklen(self):
        return self.arr.chunklen

    def __getitem__(self, item):
        with self.lock.reader():
            return self.arr.__getitem__(item)

    def __array__(self):
        with self.lock.reader():
            return self.arr[:]

    def copy(self, **kwargs):
        with self.lock.reader():
            return self.arr.copy(**kwargs)

    def __setitem__(self, key, value):
        with self.lock.writer():
            self.arr.__setitem__(key, value)

    def append(self, data):
        with self.lock.writer():
            self.arr.append(data)

    def trim(self, nitems):
        with self.lock.writer():
            self.arr.trim(nitems)

    def resize(self, nitems):
        with self.lock.writer():
            self.arr.resize(nitems)

    def reshape(self, newshape):
        with self.lock.writer():
            return self.arr.reshape(newshape)
