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

Classes (type extensions):

    carray
    earray

    __version__
"""

import sys
import numpy

_KB = 1024
_MB = 1024*_KB

# The type used for size values: indexes, coordinates, dimension
# lengths, row numbers, shapes, chunk shapes, byte counts...
SizeType = numpy.int64

__version__ = "$Revision: 4417 $"

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
                     size_t destsize)
  int blosc_decompress(void *src, void *dest, size_t destsize)
  int blosc_getitem(void *src, int start, int stop,
                    void *dest, size_t destsize)
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

# Initialize Blosc
blosc_set_nthreads(2)

#-------------------------------------------------------------


def whichLibVersion(libname):
  "Return versions of `libname` library"

  if libname == "blosc":
    return (<char *>BLOSC_VERSION_STRING, <char *>BLOSC_VERSION_DATE)


cdef class carray:
  """
  Compressed in-memory data container.

  ...blurb...

  Public instance variables
  -------------------------

  shape
  dtype

  Public methods
  --------------

  toarray()
      Get a NumPy ndarray from carray.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified by the ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object dtype
  cdef object shape
  cdef int itemsize
  cdef npy_intp nbytes, _cbytes
  cdef void *data

  property cbytes:
    """The number of compressed bytes."""
    def __get__(self):
      return SizeType(self._cbytes)


  def __cinit__(self, ndarray array, int clevel=5, int shuffle=1):
    """Initialize and compress data based on passed `array`.

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
    cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                            self.data, nbytes+BLOSC_MAX_OVERHEAD)
    if cbytes <= 0:
      raise RuntimeError, "Fatal error during Blosc compression: %d" % cbytes
    # Set size info for the instance
    self._cbytes = cbytes
    self.nbytes = nbytes
    self.itemsize = itemsize


  def toarray(self):
    """Convert this `carray` instance into a NumPy array."""
    cdef ndarray array
    cdef int ret

    # Build a NumPy container
    array = numpy.empty(shape=self.shape, dtype=self.dtype)
    # Fill it with uncompressed data
    ret = blosc_decompress(self.data, array.data, self.nbytes)
    if ret <= 0:
      raise RuntimeError, "Fatal error during Blosc decompression: %d" % ret
    return array


  cpdef _getitem(self, start, stop):
    """Read data from `start` to `stop` and return it as a NumPy array."""
    cdef ndarray array

    # Build a NumPy container
    array = numpy.empty(shape=(stop-start,), dtype=self.dtype)
    # Fill it with uncompressed data
    ret = blosc_getitem(self.data, start, stop,
                        array.data, array.size*self.itemsize)
    if ret < 0:
      raise RuntimeError, "Error in `blosc_getitem()` method."
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
    """Represent the carray as an string."""
    return str(self.toarray())


  def __repr__(self):
    """Represent the record as an string."""
    cratio = self.nbytes / float(self._cbytes)
    fullrepr = "nbytes: %d; cbytes: %d; compr. ratio: %.2f\n%r" % \
        (self.nbytes, self._cbytes, cratio, self.toarray())
    return fullrepr


  def __dealloc__(self):
    """Release C resources before destruction."""
    free(self.data)



cdef class earray:
  """
  Compressed and enlargeable in-memory data container.

  ...blurb...

  Public instance variables
  -------------------------

  shape
  dtype

  Public methods
  --------------

  toarray()
      Get a NumPy ndarray from earray.

  Special methods
  ---------------

  __getitem__(key)
      Get the values specified by the ``key``.
  __setitem__(key, value)
      Set the specified ``value`` in ``key``.
  """

  cdef object dtype, shape, chunks
  cdef int itemsize, chunksize, leftover
  cdef int clevel, shuffle
  cdef npy_intp nbytes, _cbytes
  cdef void *lastchunk
  cdef object lastchunkarr

  property cbytes:
    """The number of compressed bytes."""
    def __get__(self):
      return SizeType(self._cbytes)


  def __cinit__(self, ndarray array, int clevel=5, int shuffle=1,
                int chunksize=1*_MB):
    """Initialize and compress data based on passed `array`.

    You can pass `clevel` and `shuffle` params to the internal compressor.
    """
    cdef int i, itemsize, leftover, cs, nchunks, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef ndarray remainder, lastchunkarr

    assert len(array.shape) == 1, "Only unidimensional shapes supported."

    self.clevel = clevel
    self.shuffle = shuffle
    self.dtype = dtype = array.dtype
    self.shape = shape = array.shape
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
    for i in self.shape:
      nbytes *= i
    self.nbytes = nbytes

    # Compress data in chunks
    cbytes = 0
    nchunks = self.nbytes // self.chunksize
    nelemchunk = self.chunksize // itemsize
    for i in range(nchunks):
      chunk = carray(array[i*nelemchunk:(i+1)*nelemchunk], clevel, shuffle)
      chunks.append(chunk)
      cbytes += chunk.cbytes 
    self.leftover = leftover = nbytes % cs
    if leftover:
      remainder = array[nchunks*nelemchunk:]
      memcpy(self.lastchunk, remainder.data, leftover)
    cbytes += self.chunksize  # count the space in last chunk 
    self._cbytes = cbytes


  def toarray(self):
    """Convert this `earray` instance into a NumPy array."""
    cdef ndarray array, chunk
    cdef int ret, i, nchunks

    # Build a NumPy container
    array = numpy.empty(shape=self.shape, dtype=self.dtype)

    # Fill it with uncompressed data
    nchunks = self.nbytes // self.chunksize
    for i in range(nchunks):
      chunk = self.chunks[i].toarray()
      memcpy(array.data+i*self.chunksize, chunk.data, self.chunksize) 
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
    cdef ndarray array, chunk
    cdef int i, itemsize, chunklen, leftover, nchunks
    cdef int startb, stopb, bsize
    cdef npy_intp nbytes, ntbytes, nrows

    nbytes = self.nbytes
    itemsize = self.itemsize
    leftover = self.leftover
    chunklen = self.chunksize // itemsize
    nchunks = self.nbytes // self.chunksize
    scalar = False

    if isinstance(key, int):
      (start, stop, step) = key, key+1, 1
      scalar = True
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    else:
      raise KeyError, "key not supported:", key
    nrows = nbytes // itemsize
    start, stop, step = self._processRange(start, stop, step, nrows)

    # Build a NumPy container
    array = numpy.empty(shape=(stop-start,), dtype=self.dtype)

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
        chunk = self.chunks[i]._getitem(startb, stopb)
        memcpy(array.data+ntbytes, chunk.data, bsize)
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
    """Append `array` at the end of `self`.

    Return the number of elements appended.
    """
    cdef int itemsize, chunksize, leftover, bsize
    cdef int nbytesfirst, nelemchunk
    cdef npy_intp nbytes, cbytes
    cdef object chunk
    cdef ndarray remainder

    assert array.dtype == self.dtype, "array dtype does not match with self."
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
      chunk = carray(self.lastchunkarr, self.clevel, self.shuffle)
      chunks.append(chunk)
      cbytes = chunk.cbytes
      
      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = nbytes // chunksize
      nelemchunk = chunksize // itemsize
      # Get a new view skipping the elements that have been already copied
      remainder = array[nbytesfirst // itemsize:]
      for i in range(nchunks):
        chunk = carray(remainder[i*nelemchunk:(i+1)*nelemchunk],
                       self.clevel, self.shuffle)
        chunks.append(chunk)
        cbytes += chunk.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*nelemchunk:]
        memcpy(self.lastchunk, remainder.data, leftover)

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self.nbytes += bsize
    self.shape = (self.nbytes//itemsize)
    # Return the number of elements added
    return array.size


  def __str__(self):
    """Represent the earray as an string."""
    return str(self.toarray())


  def __repr__(self):
    """Represent the record as an string."""
    cratio = self.nbytes / float(self._cbytes)
    fullrepr = "nbytes: %d; cbytes: %d; compr. ratio: %.2f\n%r" % \
        (self.nbytes, self._cbytes, cratio, self.toarray())
    return fullrepr




## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
