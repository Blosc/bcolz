from numpy cimport ndarray, dtype, npy_intp

cdef class chunk:
    cdef public int atomsize, itemsize, blocksize
    cdef public int nbytes, cbytes
    cdef object dobject
    cdef char typekind, isconstant
    cdef int true_count
    cdef object atom, constant

    cdef void _getitem(self, int start, int stop, char *dest)
    cdef compress_data(self, char *data, size_t itemsize, size_t nbytes,
                       object cparams)
    cdef compress_arrdata(self, ndarray array, int itemsize,
                          object cparams, object _memory)


cdef class chunks(object):
    cdef object _rootdir, _mode
    cdef object dtype, cparams, lastchunkarr
    cdef object chunk_cached
    cdef npy_intp nchunks, nchunk_cached, len
    cdef int _iter_count

    cdef read_chunk(self, nchunk)
    cdef _save(self, nchunk, chunk_)
    cdef _chunk_file_name(self, nchunk)

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
    cdef object _dtype
    cdef object _safe
    cdef public object chunks
    cdef object _rootdir, datadir, metadir, _mode
    cdef object _attrs, iter_exhausted
    cdef ndarray iobuf, where_buf
    # For block cache
    cdef int idxcache
    cdef ndarray blockcache
    cdef char *datacache

    cdef void bool_update(self, boolarr, value)
    cdef int getitem_cache(self, npy_intp pos, char *dest)
    cdef reset_iter_sentinels(self)
    cdef int check_zeros(self, object barr)
    cdef _adapt_dtype(self, dtype_, shape)

