#from definitions cimport strncmp

DEF MAX_DIMS = 255 # TODO: use NumPy convention

ctypedef char * char_ptr
ctypedef void * void_ptr
ctypedef void_ptr * void_ptr_ptr

cdef struct Header_t:
    char[1]             dtype
    npy_intp            ndim
    npy_intp[MAX_DIMS]  dims

cdef struct Remainder_t:
    void*               data

cdef struct Chunks_t:
    npy_intp            numchunks
    npy_intp            chunksize
    void*               data

cdef struct StreamInfo_t:    
    Header_t            header
    Remainder_t         remain
    Chunks_t            chunks
    char[2]             streamtype
    char[1]             checkmark
    npy_intp            hsize
    npy_intp            rsize
    npy_intp            csize
    npy_intp            totsize

ctypedef StreamInfo_t StreamInfo

cdef char* CHECKMARK = '|'

cdef int SIZE_SIZE = sizeof(npy_intp)

cdef void _fill_header_info(StreamInfo* info, carray c) except *:
    cdef Header_t* header = &(info.header)
    # fill in dtype information
    header.dtype[0] = (<char*>c.dtype.char)[0]
    # fill in dimensional information
    header.ndim = c.ndim
    for i in range(header.ndim):
        header.dims[i] = c.shape[i]
    # fill in size information
    info.hsize = 1 + (1+header.ndim)*SIZE_SIZE


cdef void _fill_remain_info(StreamInfo* info, carray c) except *:
    info.rsize = c.leftover


cdef void _fill_chunk_info(StreamInfo* info, carray c) except *:
    cdef Chunks_t* chunks = &(info.chunks)
    chunks.numchunks = len(c.chunks)
    chunks.chunksize = c._chunksize
    info.csize = 2*SIZE_SIZE + chunks.chunksize*chunks.numchunks


cdef void fill_stream_info(StreamInfo* info, carray c) except *:
    _fill_header_info(info, c)
    _fill_remain_info(info, c)
    _fill_chunk_info(info, c)
    info.streamtype[0] = 'C'    # TODO: multiple type support
    info.streamtype[1] = 'A'
    info.totsize = 0            # initialize
    info.totsize += SIZE_SIZE   # total size mark
    info.totsize += 2           # streamtype mark
    info.totsize += 3           # three checkmarks
    info.totsize += 3*SIZE_SIZE # component size marks
    info.totsize += info.hsize  # header size
    info.totsize += info.rsize  # remainder size
    info.totsize += info.csize  # chunks size


cdef object create_stream(carray c):
    cdef StreamInfo info
    cdef char* dp # stream pointer
    cdef int tmp # for internal use
    cdef chunk chunk_
    
    fill_stream_info(&info, c)
    
    obj = PyBytes_FromStringAndSize(NULL, info.totsize)
    # TODO: check for NULL, MemoryError
    dp = PyBytes_AS_STRING(obj)
    
    memcpy(dp, &(info.totsize), SIZE_SIZE);          dp += SIZE_SIZE
    memcpy(dp, &(info.streamtype), 2);               dp += 2
    memcpy(dp, CHECKMARK, 1);                     dp += 1
    
    memcpy(dp, &(info.hsize), SIZE_SIZE);            dp += SIZE_SIZE
    memcpy(dp, &(info.header.dtype), 1);             dp += 1
    memcpy(dp, &(info.header.ndim), SIZE_SIZE);      dp += SIZE_SIZE
    tmp = info.header.ndim*SIZE_SIZE
    memcpy(dp, &(info.header.dims), tmp);            dp += tmp
    memcpy(dp, CHECKMARK, 1);                     dp += 1
    
    memcpy(dp, &(info.rsize), SIZE_SIZE);            dp += SIZE_SIZE
    memcpy(dp, c.lastchunk, info.rsize);             dp += info.rsize
    memcpy(dp, CHECKMARK, 1);                     dp += 1
    
    memcpy(dp, &(info.csize), SIZE_SIZE);            dp += SIZE_SIZE
    memcpy(dp, &(info.chunks.numchunks), SIZE_SIZE); dp += SIZE_SIZE
    memcpy(dp, &(info.chunks.chunksize), SIZE_SIZE); dp += SIZE_SIZE
    tmp = info.chunks.chunksize
    for chunk_ in c.chunks:
        memcpy(dp, chunk_.data, tmp);                dp += tmp
    
    return obj


cdef void assert_checkmark(void* stream, npy_intp offset) except *:
    assert (<char*>stream)[offset] == CHECKMARK[0], 'Checkmark fail!'


cdef void _extract_header_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    cdef Header_t* header = &(info.header)
    # extract header size
    info.hsize = (<npy_intp*>sp)[0];                sp += SIZE_SIZE
    assert_checkmark(stream, offset+SIZE_SIZE+info.hsize)
    # extract dtype information
    header.dtype[0] = (<char*>sp)[0];               sp += 1
    # extract dimensional information
    header.ndim = (<npy_intp*>sp)[0];               sp += SIZE_SIZE
    for n in range(header.ndim):
        header.dims[n] = (<npy_intp*>sp)[0];        sp += SIZE_SIZE


cdef void _extract_remain_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    # extract remainder size
    info.rsize = (<npy_intp*>sp)[0];                sp += SIZE_SIZE
    assert_checkmark(stream, offset+SIZE_SIZE+info.rsize)
    # hook data pointer
    info.remain.data = sp


cdef void _extract_chunk_info(StreamInfo* info, void* stream, npy_intp offset) except *:
    cdef void* sp = stream + offset
    cdef Chunks_t* chunks = &(info.chunks)
    # extract chunks size
    info.csize = (<npy_intp*>sp)[0];                sp += SIZE_SIZE
    # extract number of chunks
    chunks.numchunks = (<npy_intp*>sp)[0];          sp += SIZE_SIZE
    chunks.chunksize = (<npy_intp*>sp)[0];          sp += SIZE_SIZE
    # hook data pointer
    chunks.data = sp

cdef void extract_stream_info(StreamInfo* info, void* stream) except *:
    cdef npy_intp offset = 0
    # extract totalsize
    info.totsize = (<npy_intp*>stream)[offset];     offset += SIZE_SIZE
    # extract streamtype
    info.streamtype[0] = (<char*>stream)[offset] 			
    info.streamtype[1] = (<char*>stream)[1+offset];          offset += 2
    assert_checkmark(stream, offset);               offset += 1
       
    _extract_header_info(info, stream, offset);     offset += (SIZE_SIZE + info.hsize + 1)
    _extract_remain_info(info, stream, offset);     offset += (SIZE_SIZE + info.rsize + 1)
    _extract_chunk_info(info, stream, offset);      offset += (SIZE_SIZE + info.csize)
    
    assert offset == info.totsize, 'wrong sizes: %s != %s' % (offset, info.totsize)

cdef from_stream(char* stream):
    cdef StreamInfo info
    cdef carray carray_
    cdef ndarray chunk_
    
    extract_stream_info(&info, stream)
    
    dtype = np.dtype(info.header.dtype) # might fail.. <TEST!>
    shape = tuple([info.header.dims[n] for n in range(info.header.ndim)])
    
    cdef npy_intp numchunks = info.chunks.numchunks
    cdef npy_intp chunksize = info.chunks.chunksize
    cdef void* dp = info.chunks.data
    cdef size_t nbytes
    cdef size_t cbytes
    cdef size_t blocksize
    cdef int ret
    
    carray_ = carray(np.empty(0, dtype=dtype))
    
    for i in range(numchunks):
        blosc_cbuffer_sizes(dp, &nbytes, &cbytes, &blocksize)
        chunk_ = np.empty(nbytes / dtype.itemsize, dtype=dtype)
        ret = blosc_decompress(dp, chunk_.data, nbytes)
        if ret < 0:
            raise RuntimeError, "fatal error during Blosc decompression: %d" % ret
        carray_.append(chunk_)
        dp += chunksize
    else:
        chunk_ = np.empty(info.rsize / dtype.itemsize, dtype=dtype)
        memcpy(chunk_.data, info.remain.data, info.rsize)
        carray_.append(chunk_)
    
    if shape != carray_.shape:
        return carray_.reshape(shape) # TODO: figure out how to reshape without a copy
    return carray_
