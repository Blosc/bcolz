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

  This class is meant to be used by carray class.

  Public instance variables
  -------------------------

  Public methods
  --------------

  toarray()
      Get a numpy ndarray from chunk.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified by the ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object dtype
  cdef object shape
  cdef int itemsize, nbytes, cbytes
  cdef void *data

  def __cinit__(self, ndarray array, int clevel=5, int shuffle=1):
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
      raise RuntimeError, "Fatal error during Blosc compression: %d" % cbytes
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
      raise RuntimeError, "Fatal error during Blosc decompression: %d" % ret
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
      raise RuntimeError, "Fatal error during Blosc decompression: %d" % ret
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
  Compressed and enlargeable in-memory data container.

  This class is designed for public consumption.

  Public instance variables
  -------------------------

  shape -- the shape of this array
  dtype -- the data type of this array

  Public methods
  --------------

  toarray()
      Get a numpy array from this carray instance.

  append(array)
      Append a numpy array to this carray instance.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified by the ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object _dtype, chunks
  cdef int itemsize, chunksize, leftover
  cdef int clevel, shuffle
  cdef npy_intp nbytes, _cbytes
  cdef void *lastchunk
  cdef object lastchunkarr

  property dtype:
    """The dtype of this instance."""
    def __get__(self):
      return self._dtype

  property shape:
    """The shape of this instance."""
    def __get__(self):
      return (self.nbytes//self.itemsize,)

  property cbytes:
    """The compressed size of this array (in bytes)."""
    def __get__(self):
      return self._cbytes


  def __cinit__(self, ndarray array, int clevel=5, int shuffle=1,
                int chunksize=1*_MB):
    """Initialize and compress data based on passed `array`.

    You can pass `clevel` and `shuffle` params to the internal compressor.
    Also, you can taylor the size of the `chunksize` too.
    """
    cdef int i, itemsize, leftover, cs, nchunks, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray remainder, lastchunkarr
    cdef chunk chunk_

    assert len(array.shape) == 1, "Only unidimensional shapes supported."

    self.clevel = clevel
    self.shuffle = shuffle
    self._dtype = dtype = array.dtype
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
    for i in array.shape:
      nbytes *= i
    self.nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    nchunks = self.nbytes // self.chunksize
    nelemchunk = self.chunksize // itemsize
    for i in range(nchunks):
      chunk_ = chunk(array[i*nelemchunk:(i+1)*nelemchunk], clevel, shuffle)
      chunks.append(chunk_)
      cbytes += chunk_.cbytes 
    self.leftover = leftover = nbytes % cs
    if leftover:
      remainder = array[nchunks*nelemchunk:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self.chunksize  # count the space in last chunk 
    self._cbytes = cbytes


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


  def _processRange(self, start, stop, step, nrows, warn_negstep=True):
    """Return sensible values of start, stop and step for nrows length."""

    if warn_negstep and step and step < 0 :
      raise ValueError("slice step cannot be negative")
    # In order to convert possible numpy.integer values to long ones
    if start is not None: start = long(start)
    if stop is not None: stop = long(stop)
    if step is not None: step = long(step)
    (start, stop, step) = slice(start, stop, step).indices(nrows)

    return (start, stop, step)


  def __getitem__(self, object key):
    """__getitem__(self, key) -> values."""
    cdef ndarray array, chunk_
    cdef int i, itemsize, chunklen, leftover, nchunks
    cdef int startb, stopb, bsize
    cdef npy_intp nbytes, ntbytes, nrows

    nbytes = self.nbytes
    itemsize = self.itemsize
    leftover = self.leftover
    chunklen = self.chunksize // itemsize
    nchunks = self.nbytes // self.chunksize
    scalar = False

    # Get rid of multidimensional keys
    if isinstance(key, tuple):
      assert len(key) == 1, "Multidimensional keys are not supported"
      key = key[0]

    if isinstance(key, int):
      (start, stop, step) = key, key+1, 1
      scalar = True
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise KeyError, "key not supported: %s" % repr(key)

    nrows = nbytes // itemsize
    start, stop, step = self._processRange(start, stop, step, nrows)

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


  def append(self, ndarray array):
    """Append a numpy array to this carray instance.

    Return the number of elements appended.
    """
    cdef int itemsize, chunksize, leftover, bsize
    cdef int nbytesfirst, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray remainder
    cdef chunk chunk_

    assert array.dtype == self._dtype, "array dtype does not match with self."
    assert len(array.shape) == 1, "Only unidimensional shapes supported."

    itemsize = self.itemsize
    chunksize = self.chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = array.size*itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      memcpy(self.lastchunk+leftover, array.data, bsize)
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely
      nbytesfirst = chunksize-leftover
      memcpy(self.lastchunk+leftover, array.data, nbytesfirst)
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
    return array.size


  def __str__(self):
    """Represent the carray as an string."""
    return str(self.toarray())


  def __repr__(self):
    """Represent the carray as an string, with additional info."""
    cratio = self.nbytes / float(self._cbytes)
    array = self.toarray()
    fullrepr = "carray(%s, %s)  nbytes: %d; cbytes: %d; ratio: %.2f\n%r" % \
        (self.shape, self.dtype, self.nbytes, self._cbytes, cratio, array)
    return fullrepr




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
