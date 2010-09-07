########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
#       $Id: carrayExtension.pyx  $
#
########################################################################

"""The carray extension.

Public classes (type extensions):

    carray

Public functions:

    blosc_set_num_threads
    blosc_version

"""

import numpy as np
import carray as ca

_KB = 1024
_MB = 1024*_KB

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = np.int64

#-----------------------------------------------------------------

# numpy functions & objects
from definitions cimport import_array, ndarray, \
     malloc, free, memcpy, strdup, strcmp, \
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
  int blosc_getitem(void *src, int start, int stop,
                    void *dest, size_t destsize) nogil
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
def blosc_set_num_threads(nthreads):
  """Set the number of threads that Blosc can use.

  Returns the previous setting for this number.
  """
  return blosc_set_nthreads(nthreads)


def blosc_version():
  """Return the version of the Blosc library."""

  return (<char *>BLOSC_VERSION_STRING, <char *>BLOSC_VERSION_DATE)



cdef class chunk:
  """
  Compressed in-memory container for a data chunk.

  This class is meant to be used only by the `carray` class.

  Public methods
  --------------

  None

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified in ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object dtype
  cdef object shape
  cdef object cparms
  cdef int itemsize, nbytes, cbytes
  cdef ndarray arr1
  cdef char *data

  def __cinit__(self, ndarray array, object cparms):
    """Initialize chunk and compress data based on numpy `array`.

    You can pass parameters to the internal compressor in `cparms` that must
    be an instance of the `cparms` class.
    """
    cdef int i, itemsize
    cdef int nbytes, cbytes
    cdef int clevel, shuffle

    dtype = array.dtype
    shape = array.shape
    self.dtype = dtype
    self.shape = shape
    self.cparms = cparms

    itemsize = dtype.itemsize
    nbytes = itemsize
    for i in self.shape:
      nbytes *= i
    self.data = <char *>malloc(nbytes+BLOSC_MAX_OVERHEAD)
    # Compress data
    clevel = cparms.clevel
    shuffle = cparms.shuffle
    with nogil:
      cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                              self.data, nbytes+BLOSC_MAX_OVERHEAD)
    if cbytes <= 0:
      raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
    # Set size info for the instance
    self.cbytes = cbytes
    self.nbytes = nbytes
    self.itemsize = itemsize

    # Cache a length 1 array for accelerating getitem[int]
    self.arr1 = np.empty(shape=(1,), dtype=self.dtype)


  cdef _getitem(self, int start, int stop, char *dest):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef int bsize, ret

    bsize = (stop - start) * self.itemsize
    # Fill it with uncompressed data
    with nogil:
      if bsize == self.nbytes:
        ret = blosc_decompress(self.data, dest, bsize)
      else:
        ret = blosc_getitem(self.data, start, stop, dest, bsize)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array

    if isinstance(key, int):
      array = self.arr1
      # Quickly return a single element
      self._getitem(key, key+1, array.data)
      return PyArray_GETITEM(array, array.data)
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise IndexError, "key not supported:", key

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.shape[0])

    # Build a numpy container
    array = np.empty(shape=(stop-start,), dtype=self.dtype)
    # Read actual data
    self._getitem(start, stop, array.data)

    # Return the value depending on the key and step
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
  A compressed and enlargeable in-memory data container.

  This class is designed for public consumption.

  Public methods
  --------------

  * append(array)

  Special methods
  ---------------

  * __getitem__(key)
  * __setitem__(key, value)
  * __len__()
  * __sizeof__()

  """

  cdef int itemsize, _chunksize, leftover
  cdef int startb, stopb, nrowsinbuf, nrowsinblock, _row
  cdef int sss_mode, where_mode, getif_mode
  cdef npy_intp start, stop, step, nextelement
  cdef npy_intp _nrow, nrowsread, getif_cached
  cdef npy_intp _nbytes, _cbytes
  cdef char *lastchunk
  cdef object _cparms
  cdef object lastchunkarr
  cdef object _dtype, chunks
  cdef object getif_arr
  cdef ndarray iobuf, getif_buf

  property nrows:
    "The number of rows (leading dimension) in this carray."
    def __get__(self):
      return self._nbytes // self.itemsize

  property dtype:
    "The dtype of this carray."
    def __get__(self):
      return self._dtype

  property shape:
    "The shape of this carray."
    def __get__(self):
      return (self.nrows,)

  property cparms:
    "The compression parameters for this carray."
    def __get__(self):
      return self._cparms

  property nbytes:
    "The original (uncompressed) size of this carray (in bytes)."
    def __get__(self):
      return self._nbytes

  property cbytes:
    "The compressed size of this carray (in bytes)."
    def __get__(self):
      return self._cbytes

  property chunksize:
    "The chunksize of this carray (in bytes)."
    def __get__(self):
      return self._chunksize


  def _to_ndarray(self, object array, object arrlen=None):
    """Convert object to a ndarray."""

    if type(array) != np.ndarray:
      try:
        array = np.asarray(array, dtype=self.dtype)
      except ValueError:
        raise ValueError, "cannot convert to an ndarray object"
    # We need a contiguous array
    if not array.flags.contiguous:
      array = array.copy()
    if len(array.shape) == 0:
      # We treat scalars like undimensional arrays
      array.shape = (1,)
    if len(array.shape) != 1:
      raise ValueError, "only unidimensional shapes supported"

    # Check if we need doing a broadcast
    if (arrlen is not None) and (len(array) == 1) and (arrlen != len(array)):
      array2 = np.empty(shape=(arrlen,), dtype=array.dtype)
      array2[:] = array   # broadcast
      array = array2

    return array


  def __cinit__(self, object array, object cparms=None,
                object expectedrows=None, object chunksize=None):
    """Initialize and compress data based on passed `array`.

    You can pass parameters to the compressor via `cparms`, which must be an
    instance of the `cparms` class.

    If you pass a guess on the expected number of rows of this carray in
    `expectedrows` that wil serve to decide the best chunksize used for memory
    I/O purposes.

    Also, you can explicitely set the size of the `chunksize` used for the
    internal I/O buffer and the size of each chunk.  Only touch this if you
    know what are you doing.
    """
    cdef int i, itemsize, leftover, cs, nchunks, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray array_, remainder, lastchunkarr
    cdef chunk chunk_

    # Check defaults for cparms
    if cparms is None:
      cparms = ca.cparms()

    if not isinstance(cparms, ca.cparms):
      raise ValueError, "`cparms` param must be an instance of `cparms` class"

    array_ = self._to_ndarray(array)

    # Only accept unidimensional arrays as input
    if array_.ndim != 1:
      raise ValueError, "`array` can only be unidimensional"

    self._cparms = cparms
    self._dtype = dtype = array_.dtype
    self.chunks = chunks = []
    self.itemsize = itemsize = dtype.itemsize

    # Compute the chunksize
    if expectedrows is None:
      # Try a guess
      expectedrows = len(array_)
    if chunksize is None:
      chunksize = ca.utils.calc_chunksize(
        (expectedrows * itemsize) / float(_MB))

    # Chunksize must be a multiple of itemsize
    cs = (chunksize // itemsize) * itemsize
    self._chunksize = cs
    # Book memory for last chunk (uncompressed)
    lastchunkarr = np.empty(dtype=dtype, shape=(cs//itemsize,))
    self.lastchunk = lastchunkarr.data
    self.lastchunkarr = lastchunkarr

    # The number of bytes in incoming array
    nbytes = itemsize
    for i in array_.shape:
      nbytes *= i
    self._nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    nchunks = self._nbytes // self._chunksize
    nelemchunk = self._chunksize // itemsize
    for i in range(nchunks):
      chunk_ = chunk(array_[i*nelemchunk:(i+1)*nelemchunk], self._cparms)
      chunks.append(chunk_)
      cbytes += chunk_.cbytes
    self.leftover = leftover = nbytes % cs
    if leftover:
      remainder = array_[nchunks*nelemchunk:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self._chunksize  # count the space in last chunk
    self._cbytes = cbytes
    self.nrowsinbuf = self._chunksize // self.itemsize

    # Sentinels
    self.sss_mode = False
    self.where_mode = False
    self.getif_mode = False


  cdef _get_chunk_block_rows(self):
    """Get the number of rows in a block of the undelying Blosc chunks."""
    cdef size_t nbytes, cbytes, blocksize
    cdef chunk cbuffer

    if len(self.chunks) > 0:
      cbuffer = self.chunks[0]
      blosc_cbuffer_sizes(<void *>cbuffer.data, &nbytes, &cbytes, &blocksize)
      return blocksize // self.itemsize
    else:
      # No compressed chunks yet.  Return the size of last chunk.
      return self.nrowsinbuf


  def append(self, object array):
    """Append a numpy `array` to this carray instance.

    Return the number of elements appended.
    """
    cdef int itemsize, chunksize, leftover, bsize
    cdef int nbytesfirst, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray remainder, array_
    cdef chunk chunk_

    array_ = self._to_ndarray(array)
    if array_.dtype != self._dtype:
      raise TypeError, "array dtype does not match with self"

    itemsize = self.itemsize
    chunksize = self._chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = array_.size*itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      memcpy(self.lastchunk+leftover, array_.data, bsize)
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely
      nbytesfirst = chunksize-leftover
      memcpy(self.lastchunk+leftover, array_.data, nbytesfirst)
      # Compress the last chunk and add it to the list
      chunk_ = chunk(self.lastchunkarr, self._cparms)
      chunks.append(chunk_)
      cbytes = chunk_.cbytes

      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = nbytes // chunksize
      nelemchunk = chunksize // itemsize
      # Get a new view skipping the elements that have been already copied
      remainder = array[nbytesfirst // itemsize:]
      for i in range(nchunks):
        chunk_ = chunk(remainder[i*nelemchunk:(i+1)*nelemchunk], self._cparms)
        chunks.append(chunk_)
        cbytes += chunk_.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*nelemchunk:]
        memcpy(self.lastchunk, remainder.data, leftover)

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self._nbytes += bsize
    # Return the number of elements added
    return array_.size


  def copy(self, **kwargs):
    """Return a copy of self.

    You can pass whatever additional arguments supported by the carray
    constructor in `kwargs`.

    If `cparms` is passed, these settings will be used for the new carray.
    If not, the settings in self will be used.
    """
    cdef int itemsize, chunksize, bsize

    # Get defaults for some parameters
    cparms = kwargs.pop('cparms', self._cparms)
    expectedrows = kwargs.pop('expectedrows', self.nrows)

    # Create a new, empty carray
    ccopy = carray(np.empty(0, dtype=self.dtype),
                   cparms=cparms,
                   expectedrows=expectedrows)

    # Now copy the carray chunk by chunk
    itemsize = self.itemsize
    chunksize = self._chunksize
    bsize = chunksize // itemsize
    for i in xrange(0, self.nrows, bsize):
      ccopy.append(self[i:i+bsize])

    return ccopy


  def __len__(self):
    """Return the length of self."""
    return self.nrows


  def __sizeof__(self):
    """Return the number of bytes taken by self."""
    return self._cbytes


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array
    cdef int i, itemsize, chunklen
    cdef int startb, stopb, bsize
    cdef npy_intp nchunk, keychunk, nchunks
    cdef npy_intp nbytes, ntbytes
    cdef chunk chunk_

    nbytes = self._nbytes
    itemsize = self.itemsize
    chunklen = self._chunksize // itemsize
    nchunks = self._nbytes // self._chunksize

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, np.int_)):
      if key < 0:
        # To support negative values
        key += self.nrows
      if key >= self.nrows:
        raise IndexError, "index out of range"
      nchunk = key // chunklen
      keychunk = key % chunklen
      if nchunk == nchunks:
        array = self.lastchunkarr
        return PyArray_GETITEM(array, array.data + keychunk * itemsize)
      else:
        return self.chunks[nchunk][keychunk]
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
      return np.array([self[i] for i in key], dtype=self.dtype)
    # A boolean or integer array (case of fancy indexing)
    elif hasattr(key, "dtype"):
      if key.dtype.type == np.bool_:
        # A boolean array
        return np.fromiter(self.getif(key), dtype=self.dtype)
      elif np.issubsctype(key, np.int_):
        # An integer array
        return np.array([self[i] for i in key], dtype=self.dtype)
      else:
        raise IndexError, \
              "arrays used as indices must be of integer (or boolean) type"
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # From now on, will only deal with [start:stop:step] slices

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.nrows)

    # Build a numpy container
    array = np.empty(shape=(stop-start,), dtype=self._dtype)

    # Fill it from data in chunks
    ntbytes = 0
    for i in range(nchunks+1):
      # Compute start & stop for each block
      startb = start - i*chunklen
      stopb = stop - i*chunklen
      if (startb >= chunklen) or (stopb <= 0):
        continue
      if startb < 0:
        startb = 0
      if stopb > chunklen:
        stopb = chunklen
      bsize = (stopb - startb) * itemsize
      if i == nchunks and self.leftover:
        memcpy(array.data+ntbytes, self.lastchunk+startb*itemsize, bsize)
      else:
        # Get the data chunk
        chunk_ = self.chunks[i]
        chunk_._getitem(startb, stopb, array.data+ntbytes)
      ntbytes += bsize

    if step == 1:
      return array
    else:
      # Do a copy to get rid of unneeded data
      return array[::step].copy()


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    cdef int i, chunklen, arrlen
    cdef int nchunk, nchunks, nwritten
    cdef int startb, stopb, blen
    cdef chunk chunk_
    cdef object cdata

    # Check for integer
    # isinstance(key, int) is not enough in Cython (?)
    if isinstance(key, (int, np.int_)):
      if key < 0:
        # To support negative values
        key += self.nrows
      if key >= self.nrows:
        raise IndexError, "index out of range"
      (start, stop, step) = key, key+1, 1
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
      if step:
        if step <= 0 :
          raise NotImplementedError("step in slice can only be positive")
        if step > 1 :
          raise NotImplementedError("step > 1 in slice not supported")
    # All the rest not implemented
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.nrows)

    # Ensure that value is a numpy array with the required length
    arrlen = ca.utils.get_len_of_range(start, stop, step)
    value = self._to_ndarray(value, arrlen=arrlen)

    # Finally, update array (does not work yet!)
    #self._update(value, start, stop, step)

    nwritten = 0
    chunklen = self._chunksize // self.itemsize
    nchunks = self._nbytes // self._chunksize

    # Loop over the chunks and overwrite them from data in value
    for i in range(nchunks+1):

      # Compute start & stop for each block
      startb = start - i*chunklen
      stopb = stop - i*chunklen
      if (startb >= chunklen) or (stopb <= 0):
        continue
      if startb < 0:
        startb = 0
      if stopb > chunklen:
        stopb = chunklen
      blen = (stopb - startb)

      # Modify the data
      if i == nchunks:
        self.lastchunkarr[startb:stopb] = value[nwritten:]
      else:
        # Get the data chunk
        chunk_ = self.chunks[i]
        self._cbytes -= chunk_.cbytes
        # Get all the values there
        cdata = chunk_[:]
        # Overwrite it with data from value
        cdata[startb:stopb] = value[nwritten:nwritten+blen]
        # Replace the chunk
        chunk_ = chunk(cdata, self._cparms)
        self.chunks[i] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes

      nwritten += blen

    # Safety check
    assert (nwritten == arrlen)


  # The next is an attempt to update a carray with support for step > 1
  # But this does not work because read I/O buffer is not aligned with
  # carray chunks.  Keeping this here for later revision.
  cdef _update(self, value, npy_intp start, npy_intp stop, npy_intp step):
    """Update self with `value` array."""
    cdef npy_intp nrow, nchunk, nextelement
    cdef npy_intp nrowsread, nrowswritten, nrowsinbuf
    cdef int startb, stopb, row
    cdef int needsflush

    nrowsinbuf = self._chunksize // self.itemsize
    nrowsread = start
    nrowswritten = 0
    nrow = start - step
    nextelement = nrow + step
    startb = 0
    needsflush = False
    while nextelement < stop:
      if nextelement >= nrowsread:
        # Skip until there is interesting information
        while nextelement >= nrowsread + nrowsinbuf:
          nrowsread += nrowsinbuf
        # Compute the end for this iteration
        stopb -= nrowsread
        if stopb > nrowsinbuf:
          stopb = nrowsinbuf
        row = startb - step
        # Write back previous buffer (if needed)
        if needsflush:
          nchunk = nrowsread // nrowsinbuf
          self._flush_chunk(nchunk, iobuf)
        # Read a new data chunk
        iobuf = self[nrowsread:nrowsread+nrowsinbuf]
        nrowsread += nrowsinbuf

      row += step
      nrow = nextelement
      if row + step >= stopb:
        # Compute the start row for the next buffer
        startb = (row + step) % nrowsinbuf
      nextelement = nrow + step

      # Update the current value in I/O buffer
      iobuf[row] = value[nrowswritten]
      nrowswritten += 1
      needsflush = True

    # Write back last buffer buffer (if needed)
    if needsflush:
      nchunk = nrowsread // nrowsinbuf
      self._flush_chunk(nchunk, iobuf)

    # Safety check
    assert (nrowswritten == len(value))

    return nrowswritten


  cdef _flush_chunk(self, nchunk, iobuf):
    """Flush a chunk with the contents of iobuf """
    cdef npy_intp nchunks
    cdef chunk chunk_

    nchunks = self._nbytes // self._chunksize
    if nchunk == nchunks and self.leftover:
      # Last chunk.  Just update it.
      self.lastchunkarr[:] = iobuf
    else:
      chunk_ = self.chunks[nchunk]      # get the data chunk
      self._cbytes -= chunk_.cbytes
      chunk_ = chunk(iobuf, self._cparms)  # build the new chunk
      self.chunks[nchunk] = chunk_      # insert the new chunk
      self._cbytes += chunk_.cbytes


  def __iter__(self):
    """Iterator for traversing the data in carray."""

    if not self.sss_mode:
      self.start = 0
      self.stop = self._nbytes // self.itemsize
      self.step = 1
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    self.getif_cached = -1
    return self


  def iter(self, start=0, stop=None, step=1):
    """Iterator with `start`, `stop` and `step` bounds."""
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.nrows)
    self.sss_mode = True
    return iter(self)


  def where(self):
    """Iterator that returns indices where carray is true."""
    # Check self
    if self.dtype.type != np.bool_:
      raise ValueError, "`self` is not an array of booleans"
    self.where_mode = True
    return iter(self)


  def getif(self, boolarr):
    """Iterator that returns values where `boolarr` is true.

    `boolarr` can either be a carray or a numpy array and must be of boolean
    type.
    """
    # Check input
    if not hasattr(boolarr, "dtype"):
      raise ValueError, "`boolarr` is not an array"
    if boolarr.dtype.type != np.bool_:
      raise ValueError, "`boolarr` is not an array of booleans"
    if len(boolarr) != self.nrows:
      raise ValueError, "`boolarr` must be of the same length than ``self``"
    self.getif_mode = True
    self.getif_arr = boolarr
    # The next is not used because it seems that reading blocks is not any
    # faster than reading complete chunks, except for some cases.  For future
    # reference, the corking code is here:
    # http://github.com/FrancescAlted/carray/blob/5964a8fe0d6989c91a345f9a87016c0bbd154843/carray/carrayExtension.pyx
    self.nrowsinblock = self._get_chunk_block_rows()
    return iter(self)


  def __next__(self):
    """Return the next element in iterator."""
    cdef char *vbool
    cdef npy_intp start

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
        if self.getif_mode:
          # Read a chunk of the boolean array too
          self.getif_buf = self.getif_arr[
            self.nrowsread:self.nrowsread+self.nrowsinbuf]
        else:
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
      if self.where_mode:
        vbool = <char *>(self.iobuf.data + self._row)
        if vbool[0]:
          return self._nrow
      elif self.getif_mode:
        vbool = <char *>(self.getif_buf.data + self._row)
        if vbool[0]:
          # Check whether I/O buffer is already cached or not
          start = self.nrowsread - self.nrowsinbuf
          if start != self.getif_cached:
            self.iobuf = self[start:start+self.nrowsinbuf]
            self.getif_cached = start
          # Return the current value in I/O buffer
          return PyArray_GETITEM(
            self.iobuf, self.iobuf.data + self._row * self.itemsize)
      else:
        # Return the current value in I/O buffer
        return PyArray_GETITEM(
          self.iobuf, self.iobuf.data + self._row * self.itemsize)
    else:
      # Reset sentinels
      self.sss_mode = False
      self.where_mode = False
      self.getif_mode = False
      self.getif_arr = None
      # Reset buffers
      self.iobuf = np.empty(0, dtype=self.dtype)
      self.getif_buf = np.empty(0, dtype=np.bool_)
      raise StopIteration        # end of iteration


  def __str__(self):
    """Represent the carray as an string."""
    if self.nrows > 100:
      return "[%s, %s, %s, ..., %s, %s, %s]\n" % (self[0], self[1], self[2],
                                                  self[-3], self[-2], self[-1])
    else:
      return str(self[:])


  def __repr__(self):
    """Represent the carray as an string, with additional info."""
    cratio = self._nbytes / float(self._cbytes)
    fullrepr = """carray(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f
  cparms := %r
%s""" % (self.shape, self.dtype, self._nbytes, self._cbytes, cratio,
         self.cparms, str(self))
    return fullrepr




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
