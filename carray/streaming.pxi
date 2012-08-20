DEF MAX_DIMS = 255 # Arbitrary convention


cdef struct Header_t:
    npy_intp            hsize
    char[2]             dtype
    npy_intp            dtype_sz
    npy_intp            ndim
    npy_intp[MAX_DIMS]  dims


cdef struct Remainder_t:
    npy_intp            rsize
    void*               data


cdef struct Chunks_t:
    npy_intp            csize
    npy_intp            numck
    void*               data


cdef struct StreamInfo_t:
    Header_t            header
    Remainder_t         remain
    Chunks_t            chunks
    char[2]             streamtype
    char[1]             checkmark
    npy_intp            totsize


ctypedef StreamInfo_t StreamInfo


cdef char* CHECKMARK = '|'
cdef int SIZE_SIZE = sizeof(npy_intp)


cdef void _fill_ca_header_info(StreamInfo* info, carray c) except *:
    cdef Header_t* header = &(info.header)
    # fill in dtype information
    header.dtype[0] = (<char*>c.dtype.byteorder)[0]
    header.dtype[1] = (<char*>c.dtype.kind)[0]
    header.dtype_sz = c.dtype.itemsize
    # fill in dimensional information
    header.ndim = c.ndim
    for i in range(header.ndim):
        header.dims[i] = c.shape[i]
    # fill in size information
    info.header.hsize = 2 + SIZE_SIZE + (1+header.ndim)*SIZE_SIZE


cdef void _fill_ca_remain_info(StreamInfo* info, carray c) except *:
    info.remain.rsize = c.leftover


cdef void _fill_ca_chunk_info(StreamInfo* info, carray c) except *:
    cdef Chunks_t* chunks = &(info.chunks)
    cdef chunk chunk_
    chunks.numck = len(c.chunks)
    cdef npy_intp bsize_all = sum(chunk_.blocksize for chunk_ in c.chunks)
    info.chunks.csize = SIZE_SIZE + bsize_all + chunks.numck*SIZE_SIZE


cdef void fill_stream_info(StreamInfo* info, object c) except *:
    # initialize
    info.totsize     =  0
    info.header.hsize = 0
    info.remain.rsize = 0
    info.chunks.csize = 0
    if isinstance(c, carray):    
      info.streamtype[0] = 'C'    # TODO: multiple type support
      info.streamtype[1] = 'A'
      _fill_ca_header_info(info, c)
      _fill_ca_remain_info(info, c)
      _fill_ca_chunk_info(info, c)
    info.totsize += SIZE_SIZE         # total size mark
    info.totsize += 2                 # streamtype mark
    info.totsize += 4                 # four checkmarks
    info.totsize += 3*SIZE_SIZE       # component size marks
    info.totsize += info.header.hsize # header size
    info.totsize += info.remain.rsize # remainder size
    info.totsize += info.chunks.csize # chunks size

cdef object to_stream(object c):
    cdef char* dp
    cdef StreamInfo info
    fill_stream_info(&info, c)
    obj = PyBytes_FromStringAndSize(NULL, info.totsize) # TODO: MemoryError?
    dp = PyBytes_AS_STRING(obj)
    if isinstance(c, carray):
      _ca_serialize(&info, c, dp)
    return obj  
    

cdef object _ca_serialize(StreamInfo* info, carray c, char* dp):
    cdef npy_intp tmp # for internal use
    cdef chunk chunk_
    
    memcpy(dp, &(info.totsize), SIZE_SIZE);         dp += SIZE_SIZE
    memcpy(dp, info.streamtype, 2);                 dp += 2
    memcpy(dp, CHECKMARK, 1);                       dp += 1
    
    memcpy(dp, &(info.header.hsize), SIZE_SIZE);    dp += SIZE_SIZE
    memcpy(dp, info.header.dtype, 2);               dp += 2
    memcpy(dp, &(info.header.dtype_sz), SIZE_SIZE); dp += SIZE_SIZE
    memcpy(dp, &(info.header.ndim), SIZE_SIZE);     dp += SIZE_SIZE
    tmp = info.header.ndim*SIZE_SIZE
    memcpy(dp, info.header.dims, tmp);              dp += tmp
    memcpy(dp, CHECKMARK, 1);                       dp += 1
    
    memcpy(dp, &(info.remain.rsize), SIZE_SIZE);    dp += SIZE_SIZE
    memcpy(dp, c.lastchunk, info.remain.rsize);     dp += info.remain.rsize
    memcpy(dp, CHECKMARK, 1);                       dp += 1
    
    memcpy(dp, &(info.chunks.csize), SIZE_SIZE);    dp += SIZE_SIZE
    memcpy(dp, &(info.chunks.numck), SIZE_SIZE);    dp += SIZE_SIZE
    for chunk_ in c.chunks:
        tmp = <npy_intp>(chunk_.blocksize)
        memcpy(dp, &(tmp), SIZE_SIZE);              dp += SIZE_SIZE
        memcpy(dp, chunk_.data, tmp);               dp += tmp
    memcpy(dp, CHECKMARK, 1);                       dp += 1


cdef void assert_checkmark(void* stream, npy_intp offset) except *:
    assert (<char*>stream)[offset] == CHECKMARK[0], 'Checkmark not found at %i!' % offset


cdef void _extract_ca_header_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    cdef Header_t* header = &(info.header)
    # extract header size
    header.hsize = (<npy_intp*>sp)[0];              sp += SIZE_SIZE
    assert_checkmark(stream, offset+SIZE_SIZE+header.hsize)
    # extract dtype information
    header.dtype[0] = (<char*>sp)[0]
    header.dtype[1] = (<char*>sp)[1];               sp += 2
    header.dtype_sz = (<npy_intp*>sp)[0];           sp += SIZE_SIZE
    # extract dimensional information
    header.ndim = (<npy_intp*>sp)[0];               sp += SIZE_SIZE
    for n in range(header.ndim):
        header.dims[n] = (<npy_intp*>sp)[0];        sp += SIZE_SIZE


cdef void _extract_ca_remain_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    # extract remainder size
    info.remain.rsize = (<npy_intp*>sp)[0];         sp += SIZE_SIZE
    assert_checkmark(stream, offset+SIZE_SIZE+info.remain.rsize)
    # hook data pointer
    info.remain.data = sp


cdef void _extract_ca_chunk_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    cdef Chunks_t* chunks = &(info.chunks)
    # extract chunks size
    chunks.csize = (<npy_intp*>sp)[0];              sp += SIZE_SIZE
    assert_checkmark(stream, offset+SIZE_SIZE+chunks.csize)
    # extract number of chunks
    chunks.numck = (<npy_intp*>sp)[0];              sp += SIZE_SIZE
    # hook data pointer
    chunks.data = sp


cdef void extract_stream_info(StreamInfo* info, void* stream) except *:
    cdef npy_intp offset = 0
    # extract totalsize
    info.totsize = (<npy_intp*>stream)[offset];     offset += SIZE_SIZE
    # extract streamtype
    info.streamtype[0] = (<char*>stream)[offset]
    info.streamtype[1] = (<char*>stream)[1+offset]; offset += 2
    assert_checkmark(stream, offset);               offset += 1
    sf = _info_to_streamformat(info)
    assert sf == 'CA', "Unsupported format '%s'!" % sf
    # extract info
    _extract_ca_header_info(info, stream, offset);  offset += (SIZE_SIZE+info.header.hsize+1)
    _extract_ca_remain_info(info, stream, offset);  offset += (SIZE_SIZE+info.remain.rsize+1)
    _extract_ca_chunk_info(info, stream, offset);   offset += (SIZE_SIZE+info.chunks.csize+1)
    # sanity check
    assert offset == info.totsize, 'wrong sizes: %s != %s' % (offset, info.totsize)


cdef object _info_to_streamformat(StreamInfo* info):
    cdef char* st = '  '
    st[0] = info.streamtype[0]
    st[1] = info.streamtype[1]
    return <object>st


cdef object _info_to_dtype(StreamInfo* info):
    cdef char* dt = '  '
    dt[0] = info.header.dtype[0]
    dt[1] = info.header.dtype[1]
    dt_str = "%s%i" % (dt, info.header.dtype_sz)
    return np.dtype(dt_str)


cdef object _info_to_shape(StreamInfo* info):
    return tuple([info.header.dims[n] for n in range(info.header.ndim)])


cdef from_stream(char* stream):
    cdef StreamInfo info
    cdef carray carray_
    cdef ndarray chunk_
    
    extract_stream_info(&info, stream)
    
    dtype = _info_to_dtype(&info)
    shape = _info_to_shape(&info)
    
    cdef npy_intp bsize_buf
    cdef npy_intp num_chunks = info.chunks.numck
    cdef void* dp = info.chunks.data
    cdef size_t nbytes
    cdef size_t cbytes
    cdef size_t blocksize
    cdef int ret
    
    # TODO: cleaner implementation
    carray_ = carray(np.empty(0, dtype=dtype))
    
    for i in range(num_chunks):
        bsize_buf = (<npy_intp*>dp)[0];     dp += SIZE_SIZE
        blosc_cbuffer_sizes(dp, &nbytes, &cbytes, &blocksize)
        chunk_ = np.empty(nbytes // dtype.itemsize, dtype=dtype)
        with nogil: # release the GIL
            ret = blosc_decompress(dp, chunk_.data, nbytes)
        if ret < 0:
            raise RuntimeError, "Blosc decompression error: %i" % ret
        carray_.append(chunk_)
        dp += bsize_buf
    else:
        chunk_ = np.empty(info.remain.rsize / dtype.itemsize, dtype=dtype)
        memcpy(chunk_.data, info.remain.data, info.remain.rsize)
        carray_.append(chunk_)
    
    if shape != carray_.shape:
        return carray_.reshape(shape) # TODO: figure out how to reshape without a copy
    return carray_
