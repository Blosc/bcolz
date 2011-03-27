
DEF MAX_DIMS = 255 # TODO: use NumPy convention

ctypedef char * char_ptr
ctypedef void * void_ptr
ctypedef void_ptr * void_ptr_ptr

cdef struct Header_t:
    char*               dtype
    npy_intp            ndim
    npy_intp[MAX_DIMS]  dims

cdef struct Chunks_t:
    void_ptr_ptr        chunkarray
    npy_intp            numchunks
    npy_intp            chunksize

cdef struct StreamInfo_t:    
    Header_t            header
    Chunks_t            chunks
    char*               streamtype
    char*               checkmark
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
    header.dtype = c.dtype.char
    # fill in dimensional information
    header.ndim = c.ndim
    for i in range(header.ndim):
        header.dims[i] = c.shape[i]
    # fill in size information
    info.hsize = 1 + (1+header.ndim)*SIZE_SIZE


cdef void _fill_chunk_info(StreamInfo* info, carray c) except *:
    cdef Chunks_t* chunks = &(info.chunks)
    chunks.numchunks = len(c.chunks)
    chunks.chunksize = c._chunksize
    info.csize = chunks.chunksize*chunks.numchunks
    info.rsize = c.leftover


cdef void fill_stream_info(StreamInfo* info, carray c) except *:
    _fill_header_info(info, c)
    _fill_chunk_info(info, c)
    info.streamtype = 'CA'
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
    
    memcpy(dp, &(info.totsize), SIZE_SIZE); dp += SIZE_SIZE
    memcpy(dp, &(info.streamtype), 2);      dp += 2
    memcpy(dp, &(CHECKMARK), 1);            dp += 1
    
    memcpy(dp, &(info.hsize), SIZE_SIZE);   dp += SIZE_SIZE
    memcpy(dp, &(info.header.dtype), 1);    dp += 1
    memcpy(dp, &(info.header.ndim), SIZE_SIZE); dp += SIZE_SIZE
    tmp = info.header.ndim*SIZE_SIZE
    memcpy(dp, &(info.header.dims), tmp);   dp += tmp
    memcpy(dp, &(CHECKMARK), 1);            dp += 1
    
    memcpy(dp, &(info.rsize), SIZE_SIZE);   dp += SIZE_SIZE
    memcpy(dp, c.lastchunk, info.rsize);    dp += info.rsize
    memcpy(dp, &(CHECKMARK), 1);            dp += 1
    
    memcpy(dp, &(info.csize), SIZE_SIZE);   dp += SIZE_SIZE
    tmp = info.chunks.chunksize
    for chunk_ in c.chunks:
        memcpy(dp, chunk_.data, tmp);       dp += tmp
    
    return obj