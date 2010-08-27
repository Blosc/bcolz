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

    setBloscMaxThreads
    whichLibVersion

"""

import sys
import numpy

_KB = 1024
_MB = 1024*_KB

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = numpy.int64

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
def setBloscMaxThreads(nthreads):
  """Set the maximum number of threads that Blosc can use.

  Returns the previous setting for maximum threads.
  """
  return blosc_set_nthreads(nthreads)


def whichLibVersion(libname):
  "Return versions of `libname` library"

  if libname == "blosc":
    return (<char *>BLOSC_VERSION_STRING, <char *>BLOSC_VERSION_DATE)



cdef class chunk:
  """
  Compressed in-memory container for a data chunk.

  This class is meant to be used only by the `carray` class.

  Public methods
  --------------

  toarray()
      Get a numpy ndarray from chunk.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified in ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object dtype
  cdef object shape
  cdef int itemsize, nbytes, cbytes
  cdef void *data

  def __cinit__(self, ndarray array, int clevel=5, int shuffle=False):
    """Initialize chunk and compress data based on numpy `array`.

    You can pass `clevel` and `shuffle` params to the internal compressor.
    """
    cdef int i, itemsize
    cdef int nbytes, cbytes

    dtype = array.dtype
    shape = array.shape
    self.dtype = dtype
    self.shape = shape
      
    itemsize = dtype.itemsize
    nbytes = itemsize
    for i in self.shape:
      nbytes *= i
    self.data = malloc(nbytes+BLOSC_MAX_OVERHEAD)
    # Compress data
    with nogil:
      cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                              self.data, nbytes+BLOSC_MAX_OVERHEAD)
    if cbytes <= 0:
      raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
    # Set size info for the instance
    self.cbytes = cbytes
    self.nbytes = nbytes
    self.itemsize = itemsize


  def toarray(self):
    """Convert this `chunk` instance into a numpy array."""
    cdef ndarray array
    cdef int ret

    # Build a numpy container
    array = numpy.empty(shape=self.shape, dtype=self.dtype)
    # Fill it with uncompressed data
    with nogil:
      ret = blosc_decompress(self.data, array.data, self.nbytes)
    if ret <= 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret
    return array


  cpdef _getitem(self, int start, int stop):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef ndarray array
    cdef int bsize, ret

    # Build a numpy container
    array = numpy.empty(shape=(stop-start,), dtype=self.dtype)
    bsize = array.size * self.itemsize
    # Fill it with uncompressed data
    with nogil:
      if bsize == self.nbytes:
        ret = blosc_decompress(self.data, array.data, bsize)
      else:
        ret = blosc_getitem(self.data, start, stop, array.data, bsize)
    if ret < 0:
      raise RuntimeError, "fatal error during Blosc decompression: %d" % ret
    return array


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""

    scalar = False
    if isinstance(key, int):
      (start, stop, step) = key, key+1, 1
      scalar = True
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise KeyError, "key not supported:", key

    # Read actual data
    array = self._getitem(start, stop)

    # Return the value depending on the key and step
    if scalar:
      return array[0]
    elif step > 1:
      return array[::step]
    return array


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    raise NotImplementedError


  def __str__(self):
    """Represent the chunk as an string."""
    return str(self.toarray())


  def __repr__(self):
    """Represent the chunk as an string, with additional info."""
    cratio = self.nbytes / float(self.cbytes)
    array = self.toarray()
    fullrepr = "chunk(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
        (self.shape, self.dtype, self.nbytes, self.cbytes, cratio, array)
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

  toarray()
      Get a numpy `array` from this carray instance.

  append(array)
      Append a numpy `array` to this carray instance.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified in ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef int itemsize, chunksize, leftover
  cdef int clevel, shuffle
  cdef int startb, stopb, nrowsinbuf, _row, sss_init
  cdef npy_intp start, stop, step, nextelement, _nrow, nrowsread
  cdef npy_intp nbytes, _cbytes
  cdef void *lastchunk
  cdef object lastchunkarr
  cdef object _dtype, chunks
  cdef ndarray iobuf

  property nrows:
    """The number of rows (leading dimension) in this array."""
    def __get__(self):
      return self.nbytes // self.itemsize

  property dtype:
    """The dtype of this instance."""
    def __get__(self):
      return self._dtype

  property shape:
    """The shape of this instance."""
    def __get__(self):
      return (self.nrows,)

  property cbytes:
    """The compressed size of this array (in bytes)."""
    def __get__(self):
      return self._cbytes


  def _to_ndarray(self, object array):
    """Convert object to a ndarray."""

    if type(array) != numpy.ndarray:
      try:
        array = numpy.asarray(array)
      except ValueError:
        raise ValueError, "cannot convert to an ndarray object"
    # We need a contiguous array
    if not array.flags.contiguous:
      array = array.copy()
    if len(array.shape) != 1:
      raise ValueError, "only unidimensional shapes supported"
    return array


  def __cinit__(self, object array, int clevel=5, int shuffle=True,
                int chunksize=1*_MB):
    """Initialize and compress data based on passed `array`.

    You can pass `clevel` and `shuffle` params to the compressor.

    Also, you can taylor the size of the `chunksize` used for the internal I/O
    buffer and the size of each chunk.  Only touch this if you know what are
    you doing.
    """
    cdef int i, itemsize, leftover, cs, nchunks, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray array_, remainder, lastchunkarr
    cdef chunk chunk_

    array_ = self._to_ndarray(array)

    self.clevel = clevel
    self.shuffle = shuffle
    self._dtype = dtype = array_.dtype
    self.chunks = chunks = []
    self.itemsize = itemsize = dtype.itemsize

    # Chunksize must be a multiple of itemsize
    cs = (chunksize // itemsize) * itemsize
    self.chunksize = cs
    # Book memory for last chunk (uncompressed)
    lastchunkarr = numpy.empty(dtype=dtype, shape=(cs//itemsize,))
    self.lastchunk = lastchunkarr.data
    self.lastchunkarr = lastchunkarr

    # The number of bytes in incoming array
    nbytes = itemsize
    for i in array_.shape:
      nbytes *= i
    self.nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    nchunks = self.nbytes // self.chunksize
    nelemchunk = self.chunksize // itemsize
    for i in range(nchunks):
      chunk_ = chunk(array_[i*nelemchunk:(i+1)*nelemchunk], clevel, shuffle)
      chunks.append(chunk_)
      cbytes += chunk_.cbytes 
    self.leftover = leftover = nbytes % cs
    if leftover:
      remainder = array_[nchunks*nelemchunk:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self.chunksize  # count the space in last chunk 
    self._cbytes = cbytes
    self.nrowsinbuf = self.chunksize // self.itemsize
    self.sss_init = False  # sentinel


  def toarray(self):
    """Get a numpy array from this carray instance."""
    cdef ndarray array, chunk_
    cdef int ret, i, nchunks

    # Build a numpy container
    array = numpy.empty(shape=self.shape, dtype=self._dtype)

    # Fill it with uncompressed data
    nchunks = self.nbytes // self.chunksize
    for i in range(nchunks):
      chunk_ = self.chunks[i].toarray()
      memcpy(array.data+i*self.chunksize, chunk_.data, self.chunksize) 
    if self.leftover:
      memcpy(array.data+nchunks*self.chunksize, self.lastchunk, self.leftover)

    return array


  def __len__(self):
    """Return the length of self."""
    return self.nrows


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array, chunk_
    cdef int i, itemsize, chunklen, leftover, nchunks
    cdef int startb, stopb, bsize
    cdef npy_intp nbytes, ntbytes

    nbytes = self.nbytes
    itemsize = self.itemsize
    leftover = self.leftover
    chunklen = self.chunksize // itemsize
    nchunks = self.nbytes // self.chunksize
    scalar = False

    # Get rid of multidimensional keys
    if isinstance(key, tuple):
      if len(key) != 1:
        raise KeyError, "multidimensional keys are not supported"
      key = key[0]

    if isinstance(key, int):
      if key >= self.nrows:
        raise IndexError, "index out of range"
      if key < 0:
        # To support negative values
        key += self.nrows
      (start, stop, step) = key, key+1, 1
      scalar = True
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise NotImplementedError, "key not supported: %s" % repr(key)

    if step and step <= 0 :
      raise NotImplementedError("step in slice can only be positive")

    # Get the corrected values for start, stop, step
    (start, stop, step) = slice(start, stop, step).indices(self.nrows)

    # Build a numpy container
    array = numpy.empty(shape=(stop-start,), dtype=self._dtype)

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
      if i == nchunks and leftover:
        memcpy(array.data+ntbytes, self.lastchunk+startb*itemsize, bsize)
      else:
        # Get the data chunk
        chunk_ = self.chunks[i]._getitem(startb, stopb)
        memcpy(array.data+ntbytes, chunk_.data, bsize)
      ntbytes += bsize

    if step == 1:
      if scalar:
        return array[0]
      else:
        return array
    else:
      return array[::step]


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None."""
    raise NotImplementedError


  def __iter__(self):
    """Iterator for traversing the data in carray."""

    if not self.sss_init:
      self.start = 0
      self.stop = self.nbytes // self.itemsize
      self.step = 1
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    return self


  def iter(self, start=0, stop=None, step=1):
    """Iterator with `start`, `stop` and `step` bounds."""
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.nrows)
    self.sss_init = True
    return iter(self)


  def __next__(self):
    """Return the next element in iterator."""

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
        # Read a data chunk
        self.iobuf = self[self.nrowsread:self.nrowsread+self.nrowsinbuf]
        self.nrowsread += self.nrowsinbuf

      self._row += self.step
      self._nrow = self.nextelement
      if self._row + self.step >= self.stopb:
        # Compute the start row for the next buffer
        self.startb = (self._row + self.step) % self.nrowsinbuf
      self.nextelement = self._nrow + self.step
      # Return the current value in I/O buffer
      return PyArray_GETITEM(
        self.iobuf, self.iobuf.data + self._row * self.itemsize)
    else:
      self.sss_init = False  # reset sss_init sentinel
      raise StopIteration        # end of iteration


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
    chunksize = self.chunksize
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
      chunk_ = chunk(self.lastchunkarr, self.clevel, self.shuffle)
      chunks.append(chunk_)
      cbytes = chunk_.cbytes
      
      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = nbytes // chunksize
      nelemchunk = chunksize // itemsize
      # Get a new view skipping the elements that have been already copied
      remainder = array[nbytesfirst // itemsize:]
      for i in range(nchunks):
        chunk_ = chunk(remainder[i*nelemchunk:(i+1)*nelemchunk],
                       self.clevel, self.shuffle)
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
    self.nbytes += bsize
    # Return the number of elements added
    return array_.size


  def __str__(self):
    """Represent the carray as an string."""
    if self.nrows > 100:
      return "[%s, %s, %s... %s, %s, %s]\n" % (self[0], self[1], self[2],
                                               self[-3], self[-2], self[-1])
    else:
      return str(self.toarray())


  def __repr__(self):
    """Represent the carray as an string, with additional info."""
    cratio = self.nbytes / float(self._cbytes)
    fullrepr = "carray(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%s" % \
        (self.shape, self.dtype, self.nbytes, self._cbytes, cratio, str(self))
    return fullrepr




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
