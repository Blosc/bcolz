########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################


import numpy as np
import carray as ca
from carray import utils

_KB = 1024
_MB = 1024*_KB

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = np.int64

#-----------------------------------------------------------------

# numpy functions & objects
from definitions cimport import_array, ndarray, dtype, \
     malloc, free, memcpy, memset, strdup, strcmp, \
     PyString_AsString, PyString_FromString, \
     Py_BEGIN_ALLOW_THREADS, Py_END_ALLOW_THREADS, \
     PyArray_GETITEM, PyArray_SETITEM, \
     npy_intp

#-----------------------------------------------------------------


# Blosc routines
cdef extern from "blosc.h":

  cdef enum:
    BLOSC_MAX_OVERHEAD,
    BLOSC_VERSION_STRING,
    BLOSC_VERSION_DATE

  void blosc_get_versions(char *version_str, char *version_date)
  int blosc_set_nthreads(int nthreads)
  int blosc_compress(int clevel, int doshuffle, size_t typesize,
                     size_t nbytes, void *src, void *dest,
                     size_t destsize) nogil
  int blosc_decompress(void *src, void *dest, size_t destsize) nogil
  int blosc_getitem(void *src, int start, int nitems, void *dest) nogil
  void blosc_free_resources()
  void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                           size_t *cbytes, size_t *blocksize)
  void blosc_cbuffer_metainfo(void *cbuffer, size_t *typesize, int *flags)
  void blosc_cbuffer_versions(void *cbuffer, int *version, int *versionlz)
  void blosc_set_blocksize(size_t blocksize)


#----------------------------------------------------------------------------

# Initialization code

# The numpy API requires this function to be called before
# using any numpy facilities in an extension module.
import_array()

#-------------------------------------------------------------

# Some utilities
def _blosc_set_nthreads(nthreads):
  """
  blosc_set_nthreads(nthreads)

  Set the number of threads that Blosc can use.

  Parameters
  ----------
  nthreads : int
      The desired number of threads to use.

  Returns
  -------
  out : int
      The previous setting for the number of threads.

  """
  return blosc_set_nthreads(nthreads)


def blosc_version():
  """
  blosc_version()

  Return the version of the Blosc library.

  """
  return (<char *>BLOSC_VERSION_STRING, <char *>BLOSC_VERSION_DATE)


# This is the same than in utils.py, but works faster in extensions
cdef get_len_of_range(npy_intp start, npy_intp stop, npy_intp step):
  """Get the length of a (start, stop, step) range."""
  cdef npy_intp n

  n = 0
  if start < stop:
      n = ((stop - start - 1) // step + 1);
  return n


cdef clip_chunk(npy_intp nchunk, npy_intp chunklen,
                npy_intp start, npy_intp stop, npy_intp step):
  """Get the limits of a certain chunk based on its length."""
  cdef npy_intp startb, stopb, blen, distance

  startb = start - nchunk * chunklen
  stopb = stop - nchunk * chunklen

  # Check limits
  if (startb >= chunklen) or (stopb <= 0):
    return startb, stopb, 0   # null size
  if startb < 0:
    startb = 0
  if stopb > chunklen:
    stopb = chunklen

  # step corrections
  if step > 1:
    # Just correcting startb is enough
    distance = (nchunk * chunklen + startb) - start
    if distance % step > 0:
      startb += (step - (distance % step))
      if startb > chunklen:
        return startb, stopb, 0  # null size

  # Compute size of the clipped block
  blen = get_len_of_range(startb, stopb, step)

  return startb, stopb, blen


cdef int check_zeros(char *data, int nbytes):
  """Check whether [data, data+nbytes] is zero or not."""
  cdef int i, iszero, chunklen, leftover
  cdef size_t *sdata

  iszero = 1
  sdata = <size_t *>data
  chunklen = nbytes // sizeof(size_t)
  leftover = nbytes % sizeof(size_t)
  with nogil:
    for i from 0 <= i < chunklen:
      if sdata[i] != 0:
        iszero = 0
        break
    else:
      data += nbytes - leftover
      for i from 0 <= i < leftover:
        if data[i] != 0:
          iszero = 0
          break
  return iszero


#-------------------------------------------------------------


cdef class chunk:
  """
  chunk(array, cparams)

  Compressed in-memory container for a data chunk.

  This class is meant to be used only by the `carray` class.

  """

  # To save space, keep these variables under a minimum
  cdef char typekind, iszero
  cdef int itemsize, blocksize
  cdef int nbytes, cbytes
  cdef char *data

  property dtype:
    "The NumPy dtype for this chunk."
    def __get__(self):
      return np.dtype("%c%d" % (self.typekind, self.itemsize))

  def __cinit__(self, ndarray array, object cparams):
    cdef int itemsize, footprint
    cdef size_t nbytes, cbytes, blocksize
    cdef int clevel, shuffle
    cdef dtype dtype_

    dtype_ = array.dtype
    self.itemsize = itemsize = dtype_.elsize
    self.typekind = dtype_.kind
    footprint = 128  # the (aprox) footprint of this instance in bytes
    # Compute the total number of bytes in this array
    nbytes = itemsize * array.size
    self.itemsize = itemsize

    # Check whether incoming data is all zeros
    self.iszero = check_zeros(array.data, nbytes)
    if self.iszero:
      cbytes = 0
      blocksize = 4*1024  # use 4 KB as a cache for blocks
      # Make blocksize a multiple of itemsize
      if blocksize % itemsize > 0:
        blocksize = (blocksize // itemsize) * itemsize
    else:
      # Data is not zero, compress it
      self.data = <char *>malloc(nbytes+BLOSC_MAX_OVERHEAD)
      # Compress data
      clevel = cparams.clevel
      shuffle = cparams.shuffle
      with nogil:
        cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                                self.data, nbytes+BLOSC_MAX_OVERHEAD)
      if cbytes <= 0:
        raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
      # Set size info for the instance
      blosc_cbuffer_sizes(self.data, &nbytes, &cbytes, &blocksize)

    # Fill instance data
    self.nbytes = nbytes
    self.cbytes = cbytes + footprint
    self.blocksize = blocksize


  cdef void _getitem(self, int start, int stop, char *dest):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef int ret, bsize, blen

    blen = stop - start
    bsize = blen * self.itemsize

    if self.iszero:
      # The chunk is made of zeros, so write them
      memset(dest, 0, bsize)
      return

    # Fill dest with uncompressed data
    with nogil:
      if bsize == self.nbytes:
        ret = blosc_decompress(self.data, dest, bsize)
      else:
        ret = blosc_getitem(self.data, start, blen, dest)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array

    if isinstance(key, (int, long)):
      # Quickly return a single element
      array = np.empty(shape=(1,), dtype=self.dtype)
      self._getitem(key, key+1, array.data)
      return PyArray_GETITEM(array, array.data)
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise IndexError, "key not supported:", key

    # Get the corrected values for start, stop, step
    clen = self.nbytes // self.itemsize
    (start, stop, step) = slice(start, stop, step).indices(clen)

    # Build a numpy container
    array = np.empty(shape=(stop-start,), dtype=self.dtype)
    # Read actual data
    self._getitem(start, stop, array.data)

    # Return the value depending on the step
    if step > 1:
      return array[::step]
    return array


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    raise NotImplementedError


  def __str__(self):
    """Represent the chunk as an string."""
    return str(self[:])


  def __repr__(self):
    """Represent the chunk as an string, with additional info."""
    cratio = self.nbytes / float(self.cbytes)
    fullrepr = "chunk(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
        (self.shape, self.dtype, self.nbytes, self.cbytes, cratio, str(self))
    return fullrepr


  def __dealloc__(self):
    """Release C resources before destruction."""
    free(self.data)



cdef class carray:
  """
  carray(array, cparams=None, expectedlen=None, chunklen=None)

  A compressed and enlargeable in-memory data container.

  `carray` exposes a series of methods for dealing with the compressed
  container in a NumPy-like way.

  Parameters
  ----------
  array : an unidimensional NumPy-like object
      This is taken as the input to create the carray.  It can be any Python
      object that can be converted into a NumPy object.  The data type of
      the resulting carray will be the same as this NumPy object.
  cparams : instance of the `cparams` class, optional
      Parameters to the internal Blosc compressor.
  expectedlen : int, optional
      A guess on the expected length of this object.  This will serve to
      decide the best `chunklen` used for compression and memory I/O
      purposes.
  chunklen : int, optional
      The number of items that fits into a chunk.  By specifying it you can
      explicitely set the chunk size used for compression and memory I/O.
      Only use it if you know what are you doing.

  """

  cdef int itemsize, _chunksize, _chunklen, leftover
  cdef int nrowsinbuf, _row
  cdef int sss_mode, wheretrue_mode, where_mode
  cdef npy_intp startb, stopb
  cdef npy_intp start, stop, step, nextelement
  cdef npy_intp _nrow, nrowsread, where_cached
  cdef npy_intp _nbytes, _cbytes
  cdef char *lastchunk
  cdef object _cparams
  cdef object lastchunkarr
  cdef object _dtype, chunks
  cdef object where_arr
  cdef ndarray iobuf, where_buf
  # For block cache
  cdef int blocksize, idxcache
  cdef ndarray blockcache
  cdef char *datacache
  cdef object arr1

  property len:
    "The length (leading dimension) of this object."
    def __get__(self):
      # Important to do the cast in order to get a npy_intp result
      return self._nbytes // <npy_intp>self.itemsize

  property dtype:
    "The dtype of this object."
    def __get__(self):
      return self._dtype

  property shape:
    "The shape of this object."
    def __get__(self):
      return (self.len,)

  property cparams:
    "The compression parameters for this object."
    def __get__(self):
      return self._cparams

  property nbytes:
    "The original (uncompressed) size of this object (in bytes)."
    def __get__(self):
      return self._nbytes

  property cbytes:
    "The compressed size of this object (in bytes)."
    def __get__(self):
      return self._cbytes

  property chunklen:
    "The chunklen of this object (in rows)."
    def __get__(self):
      return self._chunklen


  def __cinit__(self, object array, object cparams=None,
                object expectedlen=None, object chunklen=None):
    cdef int i, itemsize, chunksize, leftover, nchunks
    cdef npy_intp nbytes, cbytes
    cdef ndarray array_, remainder, lastchunkarr
    cdef chunk chunk_

    # Check defaults for cparams
    if cparams is None:
      cparams = ca.cparams()

    if not isinstance(cparams, ca.cparams):
      raise ValueError, "`cparams` param must be an instance of `cparams` class"

    array_ = utils.to_ndarray(array, self.dtype)

    # Only accept unidimensional arrays as input
    if array_.ndim != 1:
      raise ValueError, "`array` can only be unidimensional"

    self._cparams = cparams
    self._dtype = dtype = array_.dtype
    self.chunks = chunks = []
    self.itemsize = itemsize = dtype.itemsize

    # Compute the chunklen/chunksize
    if expectedlen is None:
      # Try a guess
      expectedlen = len(array_)
    if chunklen is None:
      # Try a guess
      chunksize = utils.calc_chunksize((expectedlen * itemsize) / float(_MB))
      # Chunksize must be a multiple of itemsize
      chunksize = (chunksize // itemsize) * itemsize
      # Protection against large itemsizes
      if chunksize < itemsize:
        chunksize = itemsize
    else:
      if not isinstance(chunklen, int) or chunklen < 1:
        raise ValueError, "chunklen must be a positive integer"
      chunksize = chunklen * itemsize
    chunklen = chunksize  // itemsize
    self._chunksize = chunksize
    self._chunklen = chunklen

    # Book memory for last chunk (uncompressed)
    lastchunkarr = np.empty(dtype=dtype, shape=(chunklen,))
    self.lastchunk = lastchunkarr.data
    self.lastchunkarr = lastchunkarr

    # The number of bytes in incoming array
    nbytes = itemsize * array_.size
    self._nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    nchunks = nbytes // <npy_intp>chunksize
    for i from 0 <= i < nchunks:
      chunk_ = chunk(array_[i*chunklen:(i+1)*chunklen], self._cparams)
      chunks.append(chunk_)
      cbytes += chunk_.cbytes
    self.leftover = leftover = nbytes % chunksize
    if leftover:
      remainder = array_[nchunks*chunklen:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self._chunksize  # count the space in last chunk
    self._cbytes = cbytes

    # Sentinels
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.idxcache = -1       # cache not initialized

    # Cache a len-1 array for accelerating self[int] case
    self.arr1 = np.empty(shape=(1,), dtype=self.dtype)


  def append(self, object array):
    """
    append(array)

    Append a numpy `array` to this instance.

    Parameters
    ----------
    array : NumPy-like object
        The array to be appended.  Must be compatible with shape and type of
        the carray.

    Returns
    -------
    out : int
        The number of elements appended.

    """
    cdef int itemsize, chunksize, leftover, bsize
    cdef int nbytesfirst, chunklen
    cdef npy_intp nbytes, cbytes
    cdef ndarray remainder, arrcpy
    cdef chunk chunk_

    arrcpy = utils.to_ndarray(array, self.dtype)
    if arrcpy.dtype != self._dtype:
      raise TypeError, "array dtype does not match with self"

    itemsize = self.itemsize
    chunksize = self._chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = arrcpy.size*itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      memcpy(self.lastchunk+leftover, arrcpy.data, bsize)
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely
      nbytesfirst = chunksize - leftover
      memcpy(self.lastchunk+leftover, arrcpy.data, nbytesfirst)
      # Compress the last chunk and add it to the list
      chunk_ = chunk(self.lastchunkarr, self._cparams)
      chunks.append(chunk_)
      cbytes = chunk_.cbytes

      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = nbytes // <npy_intp>chunksize
      chunklen = self._chunklen
      # Get a new view skipping the elements that have been already copied
      remainder = arrcpy[nbytesfirst // itemsize:]
      for i from 0 <= i < nchunks:
        chunk_ = chunk(remainder[i*chunklen:(i+1)*chunklen], self._cparams)
        chunks.append(chunk_)
        cbytes += chunk_.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*chunklen:]
        memcpy(self.lastchunk, remainder.data, leftover)

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self._nbytes += bsize


  def copy(self, **kwargs):
    """
    copy(**kwargs)

    Return a copy of this object.

    Parameters
    ----------
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray object
        The copy of this object.

    """
    cdef object chunklen

    # Get defaults for some parameters
    cparams = kwargs.pop('cparams', self._cparams)
    expectedlen = kwargs.pop('expectedlen', self.len)

    # Create a new, empty carray
    ccopy = carray(np.empty(0, dtype=self.dtype),
                   cparams=cparams,
                   expectedlen=expectedlen,
                   **kwargs)

    # Now copy the carray chunk by chunk
    chunklen = self._chunklen
    for i from 0 <= i < self.len by chunklen:
      ccopy.append(self[i:i + chunklen])

    return ccopy


  def __len__(self):
    return self.len


  def __sizeof__(self):
    return self._cbytes


  cdef int getitem_cache(self, npy_intp pos, char *dest):
    """Get a single item from self.  It can use an internal cache.

    It returns 1 if asked `pos` can be copied to `dest`.  Else, this returns
    0.

    WARNING: Any update operation (e.g. __setitem__) *must* disable this
    cache by setting self.idxcache = -2.
    """
    cdef int ret, itemsize, blocksize, offset
    cdef int idxcache, posinbytes, blocklen
    cdef npy_intp nchunk, nchunks, chunklen
    cdef chunk chunk_

    itemsize = self.itemsize
    nchunks = self._nbytes // <npy_intp>self._chunksize
    chunklen = self._chunklen
    nchunk = pos // <npy_intp>chunklen

    # Check whether pos is in the last chunk
    if nchunk == nchunks and self.leftover:
      posinbytes = (pos % chunklen) * itemsize
      memcpy(dest, self.lastchunk + posinbytes, itemsize)
      return 1

    chunk_ = self.chunks[nchunk]
    blocksize = chunk_.blocksize
    blocklen = blocksize // itemsize

    if itemsize > blocksize:
      # This request cannot be resolved here
      return 0

    # Check whether the cache block has to be initialized
    if self.idxcache < 0:
      self.blockcache = np.empty(shape=(blocklen,), dtype=self.dtype)
      self.datacache = self.blockcache.data
      if self.idxcache == -1:
        # Absolute first time.  Add the cache size to cbytes counter.
        self._cbytes += self.blocksize

    # Check if data is cached
    idxcache = (pos // <npy_intp>blocklen) * blocklen
    if idxcache == self.idxcache:
      # Hit!
      posinbytes = (pos % blocklen) * itemsize
      memcpy(dest, self.datacache + posinbytes, itemsize)
      return 1

    # No luck. Read a complete block.
    offset = idxcache % chunklen
    chunk_._getitem(offset, offset+blocklen, self.datacache)
    # Copy the interesting bits to dest
    posinbytes = (pos % blocklen) * itemsize
    memcpy(dest, self.datacache + posinbytes, itemsize)
    # Update the cache index
    self.idxcache = idxcache
    return 1


  def __getitem__(self, object key):
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, keychunk, nchunks
    cdef npy_intp nwrow, blen
    cdef ndarray arr1
    cdef object start, stop, step

    chunklen = self._chunklen

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, long)) or isinstance(key, np.int_):
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      arr1 = self.arr1
      if self.getitem_cache(key, arr1.data):
        return PyArray_GETITEM(arr1, arr1.data)
      # Fallback action
      nchunk = key // <npy_intp>chunklen
      keychunk = key % <npy_intp>chunklen
      return self.chunks[nchunk][keychunk]
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step and step <= 0 :
        raise NotImplementedError("step in slice can only be positive")
    # Multidimensional keys
    elif isinstance(key, tuple):
      if len(key) != 1:
        raise IndexError, "multidimensional keys are not supported"
      return self[key[0]]
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        key = np.array(key, dtype=np.int_)
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      return self[key]
    # A boolean or integer array (case of fancy indexing)
    elif hasattr(key, "dtype"):
      if key.dtype.type == np.bool_:
        # A boolean array
        if len(key) != self.len:
          raise IndexError, "boolean array length must match len(self)"
        if isinstance(key, carray):
          count = sum(key.where(key))
        else:
          count = -1
        return np.fromiter(self.where(key), dtype=self.dtype, count=count)
      elif np.issubsctype(key, np.int_):
        # An integer array
        return np.array([self[i] for i in key], dtype=self.dtype)
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # An boolean expression (case of fancy indexing)
    elif type(key) is str:
      # Evaluate
      result = ca.eval(key)
      if result.dtype.type != np.bool_:
        raise IndexError, "only boolean expressions supported"
      if len(result) != self.len:
        raise IndexError, "boolean expression outcome must match len(self)"
      # Call __getitem__ again
      return self[result]
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # From now on, will only deal with [start:stop:step] slices

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy container
    blen = get_len_of_range(start, stop, step)
    array = np.empty(shape=(blen,), dtype=self._dtype)
    if blen == 0:
      # If empty, return immediately
      return array

    # Fill it from data in chunks
    nwrow = 0
    nchunks = self._nbytes // <npy_intp>self._chunksize
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks-1 and self.leftover:
        array[nwrow:nwrow+blen] = self.lastchunkarr[startb:stopb:step]
      else:
        array[nwrow:nwrow+blen] = self.chunks[nchunk][startb:stopb:step]
      nwrow += blen

    return array


  def __setitem__(self, object key, object value):
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, keychunk, nchunks
    cdef npy_intp nwrow, blen, vlen
    cdef chunk chunk_
    cdef object start, stop, step
    cdef object cdata

    # We are going to modify data.  Mark block cache as dirty.
    if self.idxcache >= 0:
      # -2 means that cbytes counter has not to be changed
      self.idxcache = -2

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, long)) or isinstance(key, np.int_):
      if key < 0:
        # To support negative values
        key += self.len
      if key >= self.len:
        raise IndexError, "index out of range"
      (start, stop, step) = key, key+1, 1
    # Slices
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step:
        if step <= 0 :
          raise NotImplementedError("step in slice can only be positive")
    # List of integers (case of fancy indexing)
    elif isinstance(key, list):
      # Try to convert to a integer array
      try:
        key = np.array(key, dtype=np.int_)
      except:
        raise IndexError, "key cannot be converted to an array of indices"
      self[key] = value
      return
    # A boolean or integer array (case of fancy indexing)
    elif hasattr(key, "dtype"):
      if key.dtype.type == np.bool_:
        # A boolean array
        if len(key) != self.len:
          raise ValueError, "boolean array length must match len(self)"
        self.bool_update(key, value)
        return
      elif np.issubsctype(key, np.int_):
        # An integer array
        value = utils.to_ndarray(value, self.dtype, arrlen=len(key))
        # XXX This could be optimised, but it works like this
        for i, item in enumerate(key):
          self[item] = value[i]
        return
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # An boolean expression (case of fancy indexing)
    elif type(key) is str:
      # Evaluate
      result = ca.eval(key)
      if result.dtype.type != np.bool_:
        raise IndexError, "only boolean expressions supported"
      if len(result) != self.len:
        raise IndexError, "boolean expression outcome must match len(self)"
      # Call __setitem__ again
      self[result] = value
      return
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.len)

    # Build a numpy object out of value
    vlen = get_len_of_range(start, stop, step)
    if vlen == 0:
      # If range is empty, return immediately
      return
    value = utils.to_ndarray(value, self.dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = self._nbytes // <npy_intp>self._chunksize
    if self.leftover > 0:
      nchunks += 1
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, blen = clip_chunk(nchunk, chunklen, start, stop, step)
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        self.lastchunkarr[startb:stopb:step] = value[nwrow:nwrow+blen]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[startb:stopb:step] = value[nwrow:nwrow+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._cparams)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)


  # This is a private function that is specific for `eval`
  def _getrange(self, npy_intp start, npy_intp blen, ndarray out):
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nwrow, stop, cblen
    cdef npy_intp schunk, echunk, nchunk, nchunks
    cdef chunk chunk_

    # Check that we are inside limits
    nrows = self._nbytes // <npy_intp>self.itemsize
    if (start + blen) > nrows:
      blen = nrows - start

    # Fill `out` from data in chunks
    nwrow = 0
    stop = start + blen
    nchunks = self._nbytes // <npy_intp>self._chunksize
    chunklen = self._chunksize // self.itemsize
    schunk = start // <npy_intp>chunklen
    echunk = (start+blen) // <npy_intp>chunklen
    for nchunk from schunk <= nchunk <= echunk:
      # Compute start & stop for each block
      startb = start % chunklen
      stopb = chunklen
      if (start + startb) + chunklen > stop:
        # XXX I still have to explain why this expression works
        # for chunklen > (start + blen)
        stopb = (stop - start) + startb
      cblen = stopb - startb
      if cblen == 0:
        continue
      # Get the data chunk and assign it to result array
      if nchunk == nchunks and self.leftover:
        out[nwrow:nwrow+cblen] = self.lastchunkarr[startb:stopb]
      else:
        chunk_ = self.chunks[nchunk]
        chunk_._getitem(startb, stopb, out.data+nwrow*self.itemsize)
      nwrow += cblen
      start += cblen


  cdef void bool_update(self, boolarr, value):
    """Update self in positions where `boolarr` is true with `value` array."""
    cdef int chunklen
    cdef npy_intp startb, stopb
    cdef npy_intp nchunk, nchunks, nrows
    cdef npy_intp nwrow, blen, vlen, n
    cdef chunk chunk_
    cdef object cdata, boolb

    vlen = sum(boolarr)   # number of true values in bool array
    value = utils.to_ndarray(value, self.dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = self._nbytes // <npy_intp>self._chunksize
    if self.leftover > 0:
      nchunks += 1
    nrows = self._nbytes // <npy_intp>self.itemsize
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, _ = clip_chunk(nchunk, chunklen, 0, nrows, 1)
      # Get boolean values for this chunk
      n = nchunk * chunklen
      boolb = boolarr[n+startb:n+stopb]
      blen = sum(boolb)
      if blen == 0:
        continue
      # Modify the data in chunk
      if nchunk == nchunks-1 and self.leftover:
        self.lastchunkarr[boolb] = value[nwrow:nwrow+blen]
      else:
        # Get the data chunk
        chunk_ = self.chunks[nchunk]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[boolb] = value[nwrow:nwrow+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._cparams)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)


  def __iter__(self):

    if not self.sss_mode:
      self.start = 0
      self.stop = self._nbytes // <npy_intp>self.itemsize
      self.step = 1
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    self.where_cached = -1
    if self.where_mode and isinstance(self.where_arr, carray):
      self.nrowsinbuf = self.where_arr.chunklen
    else:
      self.nrowsinbuf = self._chunklen

    return self


  def iter(self, start=0, stop=None, step=1):
    """
    iter(start=0, stop=None, step=1)

    Iterator with `start`, `stop` and `step` bounds.

    Parameters
    ----------
    start : int
        The starting item.
    stop : int
        The item after which the iterator stops.
    step : int
        The number of items incremented during each iteration.  Cannot be
        negative.

    Returns
    -------
    out : iterator

    """
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.len)
    self.reset_sentinels()
    self.sss_mode = True
    return iter(self)


  def wheretrue(self):
    """
    wheretrue()

    Iterator that returns indices where this object is true.  Only useful for
    boolean carrays.

    Returns
    -------
    out : iterator

    See Also
    --------
    where

    """
    # Check self
    if self.dtype.type != np.bool_:
      raise ValueError, "`self` is not an array of booleans"
    self.reset_sentinels()
    self.wheretrue_mode = True
    return iter(self)


  def where(self, boolarr):
    """
    where(boolarr)

    Iterator that returns values of this object where `boolarr` is true.

    Parameters
    ----------
    boolarr : a carray or NumPy array of boolean type

    Returns
    -------
    out : iterator

    See also
    --------
    wheretrue

    """
    # Check input
    if not hasattr(boolarr, "dtype"):
      raise ValueError, "`boolarr` is not an array"
    if boolarr.dtype.type != np.bool_:
      raise ValueError, "`boolarr` is not an array of booleans"
    if len(boolarr) != self.len:
      raise ValueError, "`boolarr` must be of the same length than ``self``"
    self.reset_sentinels()
    self.where_mode = True
    self.where_arr = boolarr
    return iter(self)


  def __next__(self):
    cdef char *vbool
    cdef npy_intp start, nchunk

    self.nextelement = self._nrow + self.step
    while self.nextelement < self.stop:
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread += self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step
        if self.where_mode:
          # Skip chunks with zeros (false values)
          if self.check_zeros(self.where_arr):
            self.nrowsread += self.nrowsinbuf
            self.nextelement += self.nrowsinbuf
            continue
          # Read a chunk of the boolean array
          self.where_buf = self.where_arr[
            self.nrowsread:self.nrowsread+self.nrowsinbuf]
        else:
          # Skip chunks with zeros only if in wheretrue_mode
          if self.wheretrue_mode and self.check_zeros(self):
            self.nrowsread += self.nrowsinbuf
            self.nextelement += self.nrowsinbuf
            continue
          # Read a data chunk
          self.iobuf = self[self.nrowsread:self.nrowsread+self.nrowsinbuf]
        self.nrowsread += self.nrowsinbuf

      self._row += self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf
      self.nextelement = self._nrow + self.step

      # Return a value depending on the mode we are
      if self.wheretrue_mode:
        vbool = <char *>(self.iobuf.data + self._row)
        if vbool[0]:
          return self._nrow
      elif self.where_mode:
        vbool = <char *>(self.where_buf.data + self._row)
        if vbool[0]:
          # Check whether I/O buffer is already cached or not
          start = self.nrowsread - self.nrowsinbuf
          if start != self.where_cached:
            self.iobuf = self[start:start+self.nrowsinbuf]
            self.where_cached = start
          # Return the current value in I/O buffer
          return PyArray_GETITEM(
            self.iobuf, self.iobuf.data + self._row * self.itemsize)
      else:
        # Return the current value in I/O buffer
        return PyArray_GETITEM(
          self.iobuf, self.iobuf.data + self._row * self.itemsize)
    else:
      # Release buffers
      self.iobuf = np.empty(0, dtype=self.dtype)
      self.where_buf = np.empty(0, dtype=np.bool_)
      raise StopIteration        # end of iteration


  cdef reset_sentinels(self):
    """Reset sentinels for iterator."""
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.where_arr = None


  cdef int check_zeros(self, object barr):
    """Check for zeros.  Return 1 if all zeros, else return 0."""
    cdef int bsize
    cdef carray carr
    cdef ndarray ndarr
    cdef chunk chunk_

    if isinstance(barr, carray):
      # Check for zero'ed chunks in carrays
      carr = barr
      nchunk = self.nrowsread // <npy_intp>self.nrowsinbuf
      if nchunk < len(carr.chunks):
        chunk_ = carr.chunks[nchunk]
        if chunk_.iszero:
          return 1
    else:
      # Check for zero'ed chunks in ndarrays
      ndarr = barr
      bsize = self.nrowsinbuf
      if self.nrowsread + bsize > self.len:
        bsize = self.len - self.nrowsread
      if check_zeros(ndarr.data + self.nrowsread, bsize):
        return 1
    return 0


  def __str__(self):
    if self.len > 100:
      return "[%s, %s, %s, ..., %s, %s, %s]\n" % (self[0], self[1], self[2],
                                                  self[-3], self[-2], self[-1])
    else:
      return str(self[:])


  def __repr__(self):
    snbytes = utils.human_readable_size(self._nbytes)
    scbytes = utils.human_readable_size(self._cbytes)
    cratio = self._nbytes / float(self._cbytes)
    fullrepr = """carray(%s, %s)  nbytes: %s; cbytes: %s; ratio: %.2f
  cparams := %r
%s""" % (self.shape, self.dtype, snbytes, scbytes, cratio,
         self.cparams, str(self))
    return fullrepr




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
