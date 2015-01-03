# numpy functions & objects
from definitions cimport import_array, \
    malloc, realloc, free, memcpy, memset, strdup, strcmp, \
    PyString_AsString, PyString_GET_SIZE, \
    PyString_FromStringAndSize, \
    Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
    PyArray_GETITEM, PyArray_SETITEM, \
    PyBuffer_FromMemory, Py_uintptr_t

from numpy cimport ndarray, dtype, npy_intp

cdef class chunk:
    cdef char typekind, isconstant
    cdef public int atomsize, itemsize, blocksize
    cdef public int nbytes, cbytes, cdbytes
    cdef int true_count
    cdef char *data
    cdef object atom, constant, dobject

    cdef void _getitem(self, int start, int stop, char *dest)
    cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                       object cparams)
    cdef object compress_arrdata(self, ndarray array, int itemsize, object cparams, object _memory)

cdef class carray:
    cdef public int itemsize, atomsize
    cdef int _chunksize, _chunklen, leftover
    cdef int nrowsinbuf, _row
    cdef int sss_mode, wheretrue_mode, where_mode
    cdef npy_intp startb, stopb
    cdef npy_intp start, stop, step, nextelement
    cdef npy_intp _nrow, nrowsread
    cdef npy_intp _nbytes, _cbytes
    cdef npy_intp nhits, limit, skip
    cdef npy_intp expectedlen
    cdef char *lastchunk
    cdef object lastchunkarr, where_arr, arr1
    cdef object _cparams, _dflt
    cdef dtype _dtype
    cdef public object chunks
    cdef object _rootdir, datadir, metadir, _mode
    cdef object _attrs, iter_exhausted
    cdef ndarray iobuf, where_buf
    # For block cache
    cdef int idxcache
    cdef ndarray blockcache
    cdef char *datacache

    cdef _adapt_dtype(self, dtype dtype_, object shape)
    cdef int getitem_cache(self, npy_intp pos, char *dest)
    cdef void bool_update(self, boolarr, value)
    cdef reset_iter_sentinels(self)
    cdef int check_zeros(self, object barr)
