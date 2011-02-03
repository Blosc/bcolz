########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - faltet@pytables.org
#
########################################################################


import sys
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
     malloc, realloc, free, memcpy, memset, strdup, strcmp, \
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

  Sets the number of threads that Blosc can use.

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
  cdef char typekind, isconstant
  cdef int atomsize, itemsize, blocksize
  cdef int nbytes, cbytes
  cdef char *data
  cdef object atom, constant

  property dtype:
    "The NumPy dtype for this chunk."
    def __get__(self):
      return self.atom


  def __cinit__(self, ndarray array, object atom, object cparams):
    cdef int itemsize, footprint
    cdef size_t nbytes, cbytes, blocksize
    cdef int clevel, shuffle
    cdef dtype dtype_
    cdef char *dest

    self.atom = atom
    self.atomsize = atom.itemsize
    dtype_ = array.dtype
    self.itemsize = itemsize = dtype_.elsize
    self.typekind = dtype_.kind
    # Compute the total number of bytes in this array
    nbytes = itemsize * array.size
    footprint = 128  # the (aprox) footprint of this instance in bytes

    # Check whether incoming data is constant
    if array.strides[0] == 0 or check_zeros(array.data, nbytes):
      self.isconstant = 1
      self.constant = constant = array[0]
      # Add overhead (64 bytes for the overhead of the numpy container)
      footprint += 64 + constant.size * constant.itemsize
    if self.isconstant:
      cbytes = 0
      blocksize = 4*1024  # use 4 KB as a cache for blocks
      # Make blocksize a multiple of itemsize
      if blocksize % itemsize > 0:
        blocksize = (blocksize // itemsize) * itemsize
      # Correct in case we have a large itemsize
      if blocksize == 0:
        blocksize = itemsize
    else:
      # Data is not constant, compress it
      dest = <char *>malloc(nbytes+BLOSC_MAX_OVERHEAD)
      # Compress data
      clevel = cparams.clevel
      shuffle = cparams.shuffle
      with nogil:
        cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                                dest, nbytes+BLOSC_MAX_OVERHEAD)
      if cbytes <= 0:
        raise RuntimeError, "fatal error during Blosc compression: %d" % cbytes
      # Free the unused data
      # self.data = <char *>realloc(dest, cbytes)
      # I think the next is safer (and the speed is barely the same)
      # Copy the compressed data on a new tailored buffer
      self.data = <char *>malloc(cbytes)
      memcpy(self.data, dest, cbytes)
      free(dest)
      # Set size info for the instance
      blosc_cbuffer_sizes(self.data, &nbytes, &cbytes, &blocksize)

    # Fill instance data
    self.nbytes = nbytes
    self.cbytes = cbytes + footprint
    self.blocksize = blocksize


  cdef void _getitem(self, int start, int stop, char *dest):
    """Read data from `start` to `stop` and return it as a numpy array."""
    cdef int ret, bsize, blen
    cdef ndarray constants

    blen = stop - start
    bsize = blen * self.atomsize

    if self.isconstant:
      # The chunk is made of constants
      constants = np.ndarray(shape=(blen,), dtype=self.dtype,
                             buffer=self.constant, strides=(0,)).copy()
      memcpy(dest, constants.data, bsize)
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
    cdef object start, stop, step, clen, idx

    if isinstance(key, (int, long)):
      # Quickly return a single element
      array = np.empty(shape=(1,), dtype=self.dtype)
      self._getitem(key, key+1, array.data)
      return PyArray_GETITEM(array, array.data)
    elif isinstance(key, slice):
      (start, stop, step) = key.start, key.stop, key.step
    elif isinstance(key, tuple) and self.dtype.shape != ():
      # Build an array to guess indices
      clen = self.nbytes // self.itemsize
      idx = np.arange(clen, dtype=np.int32).reshape(self.dtype.shape)
      idx2 = idx(key)
      if idx2.flags.contiguous:
        # The slice represents a contiguous slice.  Get start and stop.
        start, stop = idx2.flatten()[[0,-1]]
        step = 1
      else:
        (start, stop, step) = key[0].start, key[0].stop, key[0].step
    else:
      raise IndexError, "key not suitable:", key

    # Get the corrected values for start, stop, step
    clen = self.nbytes // self.atomsize
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
  carray(array, cparams=None, dtype=None, dflt=None, expectedlen=None, chunklen=None)

  A compressed and enlargeable in-memory data container.

  `carray` exposes a series of methods for dealing with the compressed
  container in a NumPy-like way.

  Parameters
  ----------
  array : a NumPy-like object
      This is taken as the input to create the carray.  It can be any Python
      object that can be converted into a NumPy object.  The data type of
      the resulting carray will be the same as this NumPy object.
  cparams : instance of the `cparams` class, optional
      Parameters to the internal Blosc compressor.
  dtype : NumPy dtype
      Force this `dtype` for the carray (rather than the `array` one).
  dflt : Python or NumPy scalar
      The value to be used when enlarging the carray.  If None, the default is
      filling with zeros.
  expectedlen : int, optional
      A guess on the expected length of this object.  This will serve to
      decide the best `chunklen` used for compression and memory I/O
      purposes.
  chunklen : int, optional
      The number of items that fits into a chunk.  By specifying it you can
      explicitely set the chunk size used for compression and memory I/O.
      Only use it if you know what are you doing.

  """

  cdef int itemsize, atomsize, _chunksize, _chunklen, leftover
  cdef int nrowsinbuf, _row
  cdef int sss_mode, wheretrue_mode, where_mode
  cdef npy_intp startb, stopb
  cdef npy_intp start, stop, step, nextelement
  cdef npy_intp _nrow, nrowsread
  cdef npy_intp _nbytes, _cbytes
  cdef npy_intp nhits, limit
  cdef char *lastchunk
  cdef object lastchunkarr, where_arr, arr1
  cdef object _cparams, _dflt
  cdef object _dtype, chunks
  cdef ndarray iobuf, where_buf
  # For block cache
  cdef int blocksize, idxcache
  cdef ndarray blockcache
  cdef char *datacache

  property cbytes:
    "The compressed size of this object (in bytes)."
    def __get__(self):
      return self._cbytes

  property chunklen:
    "The chunklen of this object (in rows)."
    def __get__(self):
      return self._chunklen

  property cparams:
    "The compression parameters for this object."
    def __get__(self):
      return self._cparams

  property dflt:
    "The default value of this object."
    def __get__(self):
      return self._dflt

  property dtype:
    "The dtype of this object."
    def __get__(self):
      return self._dtype

  property len:
    "The length (leading dimension) of this object."
    def __get__(self):
      # Important to do the cast in order to get a npy_intp result
      return self._nbytes // <npy_intp>self.atomsize

  property nbytes:
    "The original (uncompressed) size of this object (in bytes)."
    def __get__(self):
      return self._nbytes

  property shape:
    "The shape of this object."
    def __get__(self):
      return (self.len,)


  def __cinit__(self, object array, object cparams=None,
                object dtype=None, object dflt=None,
                object expectedlen=None, object chunklen=None):
    cdef int i, itemsize, atomsize, chunksize, leftover, nchunks
    cdef npy_intp nbytes, cbytes
    cdef ndarray array_, remainder, lastchunkarr
    cdef chunk chunk_
    cdef object _dflt

    # Check defaults for cparams
    if cparams is None:
      cparams = ca.cparams()

    if not isinstance(cparams, ca.cparams):
      raise ValueError, "`cparams` param must be an instance of `cparams` class"

    # Convert input to an appropriate type
    if type(dtype) is str:
        dtype = np.dtype(dtype)
    array_ = utils.to_ndarray(array, dtype)
    if dtype is None:
      if len(array_.shape) == 1:
        self._dtype = dtype = array_.dtype
      else:
        # Multidimensional array.  The atom will have array_.shape[1:] dims.
        self._dtype = dtype = np.dtype((array_.dtype.base, array_.shape[1:]))
    else:
      self._dtype = dtype
    # Checks for the dtype
    if self._dtype.kind == 'O':
      raise TypeError, "object dtypes are not supported in carray objects"
    # Check that atom size is less than 2 GB
    if dtype.itemsize >= 2**31:
      raise ValueError, "atomic size is too large (>= 2 GB)"

    # Check defaults for dflt
    _dflt = np.zeros((), dtype=dtype)
    if dflt is not None:
      if dtype.shape == ():
        _dflt[()] = dflt
      else:
        _dflt[:] = dflt
    self._dflt = _dflt

    self._cparams = cparams
    self.chunks = chunks = []
    self.atomsize = atomsize = dtype.itemsize
    self.itemsize = itemsize = dtype.base.itemsize

    # Compute the chunklen/chunksize
    if expectedlen is None:
      # Try a guess
      expectedlen = len(array_)
    if chunklen is None:
      # Try a guess
      chunksize = utils.calc_chunksize((expectedlen * atomsize) / float(_MB))
      # Chunksize must be a multiple of atomsize
      chunksize = (chunksize // atomsize) * atomsize
      # Protection against large itemsizes
      if chunksize < atomsize:
        chunksize = atomsize
    else:
      if not isinstance(chunklen, int) or chunklen < 1:
        raise ValueError, "chunklen must be a positive integer"
      chunksize = chunklen * atomsize
    chunklen = chunksize  // atomsize
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
      chunk_ = chunk(array_[i*chunklen:(i+1)*chunklen], dtype, cparams)
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

    """
    cdef int atomsize, itemsize, chunksize, leftover
    cdef int nbytesfirst, chunklen, start, stop
    cdef npy_intp nbytes, cbytes, bsize
    cdef ndarray remainder, arrcpy, dflts
    cdef chunk chunk_

    arrcpy = utils.to_ndarray(array, self.dtype)
    if arrcpy.dtype != self._dtype.base:
      raise TypeError, "array dtype does not match with self"
    # Appending a single row should be supported
    if arrcpy.shape == self._dtype.shape:
      arrcpy = arrcpy.reshape((1,)+arrcpy.shape)
    if arrcpy.shape[1:] != self._dtype.shape:
      raise ValueError, "array trailing dimensions does not match with self"

    atomsize = self.atomsize
    itemsize = self.itemsize
    chunksize = self._chunksize
    chunks = self.chunks
    leftover = self.leftover
    bsize = arrcpy.size*itemsize
    cbytes = 0

    # Check if array fits in existing buffer
    if (bsize + leftover) < chunksize:
      # Data fits in lastchunk buffer.  Just copy it
      if arrcpy.strides[0] > 0:
        memcpy(self.lastchunk+leftover, arrcpy.data, bsize)
      else:
        start, stop = leftover // atomsize, (leftover+bsize) // atomsize
        self.lastchunkarr[start:stop] = arrcpy
      leftover += bsize
    else:
      # Data does not fit in buffer.  Break it in chunks.

      # First, fill the last buffer completely (if needed)
      if leftover:
        nbytesfirst = chunksize - leftover
        if arrcpy.strides[0] > 0:
          memcpy(self.lastchunk+leftover, arrcpy.data, nbytesfirst)
        else:
          start, stop = leftover // atomsize, (leftover+nbytesfirst) // atomsize
          self.lastchunkarr[start:stop] = arrcpy[start:stop]
        # Compress the last chunk and add it to the list
        chunk_ = chunk(self.lastchunkarr, self._dtype, self._cparams)
        chunks.append(chunk_)
        cbytes = chunk_.cbytes
      else:
        nbytesfirst = 0

      # Then fill other possible chunks
      nbytes = bsize - nbytesfirst
      nchunks = nbytes // <npy_intp>chunksize
      chunklen = self._chunklen
      # Get a new view skipping the elements that have been already copied
      remainder = arrcpy[nbytesfirst // atomsize:]
      for i from 0 <= i < nchunks:
        chunk_ = chunk(
          remainder[i*chunklen:(i+1)*chunklen], self._dtype, self._cparams)
        chunks.append(chunk_)
        cbytes += chunk_.cbytes

      # Finally, deal with the leftover
      leftover = nbytes % chunksize
      if leftover:
        remainder = remainder[nchunks*chunklen:]
        if arrcpy.strides[0] > 0:
          memcpy(self.lastchunk, remainder.data, leftover)
        else:
          self.lastchunkarr[:len(remainder)] = remainder

    # Update some counters
    self.leftover = leftover
    self._cbytes += cbytes
    self._nbytes += bsize


  def trim(self, object nitems):
    """
    trim(nitems)

    Remove the trailing `nitems` from this instance.

    Parameters
    ----------
    nitems : int
        The number of trailing items to be trimmed.  If negative, the object
        is enlarged instead.

    """
    cdef int atomsize, leftover, leftover2
    cdef npy_intp cbytes, bsize, nchunk2
    cdef chunk chunk_

    if not isinstance(nitems, (int, long, float)):
      raise TypeError, "`nitems` must be an integer"

    # Check that we don't run out of space
    if nitems > self.len:
      raise ValueError, "`nitems` must be less than total length"
    # A negative number of items means that we want to grow the object
    if nitems <= 0:
      self.resize(self.len - nitems)
      return

    atomsize = self.atomsize
    chunks = self.chunks
    leftover = self.leftover
    bsize = nitems * atomsize
    cbytes = 0


    # Check if items belong in last chunk
    if (leftover - bsize) > 0:
      # Just update leftover counter
      leftover -= bsize
    else:
      # nitems larger than last chunk
      nchunk = (self.len - nitems) // self._chunklen
      leftover2 = (self.len - nitems) % self._chunklen
      leftover = leftover2 * atomsize

      # Remove complete chunks
      nchunk2 = self._nbytes // <npy_intp>self._chunksize
      while nchunk2 > nchunk+1:
        chunk_ = chunks.pop()
        cbytes += chunk_.cbytes
        nchunk2 -= 1

      # Finally, deal with the leftover
      if leftover:
        chunk_ = chunks.pop()
        cbytes += chunk_.cbytes
        self.lastchunkarr[:leftover2] = chunk_[:leftover2]

    # Update some counters
    self.leftover = leftover
    self._cbytes -= cbytes
    self._nbytes -= bsize


  def resize(self, object nitems):
    """
    resize(nitems)

    Resize the instance to have `nitems`.

    Parameters
    ----------
    nitems : int
        The final length of the object.  If `nitems` is larger than the actual
        length, new items will appended using `self.dflt` as filling values.

    """
    cdef object chunk

    if not isinstance(nitems, (int, long, float)):
      raise TypeError, "`nitems` must be an integer"

    if nitems == self.len:
      return
    elif nitems < 0:
      raise ValueError, "`nitems` cannot be negative"

    if nitems > self.len:
      # Create a 0-strided array and append it to self
      chunk = np.ndarray(nitems-self.len, dtype=self.dtype,
                         buffer=self._dflt, strides=(0,))
      self.append(chunk)
    else:
      # Just trim the excess of items
      self.trim(self.len-nitems)


  def reshape(self, newshape):
    """
    reshape(newshape)

    Returns a new carray containing the same data with a new shape.

    Parameters
    ----------
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : carray
        A copy of the original carray.

    """
    cdef npy_intp newlen, ilen, isize, osize, newsize, rsize, i
    cdef object ishape, oshape, pos, newdtype, out

    # Enforce newshape as tuple
    if isinstance(newshape, (int, long)):
      newshape = (newshape,)
    newsize = np.prod(newshape)

    ishape = self.shape+self._dtype.shape
    ilen = ishape[0]
    isize = np.prod(ishape)

    # Check for -1 in newshape
    if -1 in newshape:
      if newshape.count(-1) > 1:
        raise ValueError, "only one shape dimension can be -1"
      pos = newshape.index(-1)
      osize = np.prod(newshape[:pos] + newshape[pos+1:])
      if isize == 0:
        newshape = newshape[:pos] + (0,) + newshape[pos+1:]
      else:
        newshape = newshape[:pos] + (isize/osize,) + newshape[pos+1:]
      newsize = np.prod(newshape)

    # Check shape compatibility
    if isize != newsize:
      raise ValueError, "`newshape` is not compatible with the current one"
    # Create the output container
    newdtype = np.dtype((self._dtype.base, newshape[1:]))
    newlen = newshape[0]

    # If shapes are both n-dimensional, convert first to 1-dim shape
    # and then convert again to the final newshape.
    if len(ishape) > 1 and len(newshape) > 1:
      out = self.reshape(-1)
      return out.reshape(newshape)

    # Create the final container and fill it
    out = carray([], dtype=newdtype, cparams=self.cparams, expectedlen=newlen)
    if newlen < ilen:
      rsize = isize / newlen
      for i from 0 <= i < newlen:
        out.append(self[i*rsize:(i+1)*rsize].reshape(newdtype.shape))
    else:
      for i from 0 <= i < ilen:
        out.append(self[i].reshape(-1))

    return out


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
      ccopy.append(self[i:i+chunklen])

    return ccopy


  def sum(self, dtype=None):
    """
    sum(dtype=None)

    Return the sum of the array elements.

    Parameters
    ----------
    dtype : NumPy dtype
        The desired type of the output.  If ``None``, the dtype of `self` is
        used.

    Return value
    ------------
    out : NumPy scalar with `dtype`

    """
    cdef chunk chunk_
    cdef object result
    cdef npy_intp nchunk, nchunks

    if dtype is None:
      dtype = self.dtype
    if dtype.kind == 'S':
      raise TypeError, "cannot perform reduce with flexible type"

    # Get a container for the result
    result = np.zeros(1, dtype=dtype)[0]

    nchunks = self._nbytes // <npy_intp>self._chunksize
    for nchunk from 0 <= nchunk < nchunks:
      chunk_ = self.chunks[nchunk]
      if chunk_.isconstant:
        # Multiplying 0's can be costly (!), so remove the need to do so
        if chunk_.constant != 0:
          result += chunk_.constant * self._chunklen
      else:
        result += chunk_[:].sum()
    if self.leftover:
      result += self.lastchunkarr[:self.len-nchunks*self._chunklen].sum()

    return result


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
    cdef int ret, atomsize, blocksize, offset
    cdef int idxcache, posinbytes, blocklen
    cdef npy_intp nchunk, nchunks, chunklen
    cdef chunk chunk_

    atomsize = self.atomsize
    nchunks = self._nbytes // <npy_intp>self._chunksize
    chunklen = self._chunklen
    nchunk = pos // <npy_intp>chunklen

    # Check whether pos is in the last chunk
    if nchunk == nchunks and self.leftover:
      posinbytes = (pos % chunklen) * atomsize
      memcpy(dest, self.lastchunk + posinbytes, atomsize)
      return 1

    chunk_ = self.chunks[nchunk]
    blocksize = chunk_.blocksize
    blocklen = blocksize // atomsize

    if atomsize > blocksize:
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
      posinbytes = (pos % blocklen) * atomsize
      memcpy(dest, self.datacache + posinbytes, atomsize)
      return 1

    # No luck. Read a complete block.
    offset = idxcache % chunklen
    chunk_._getitem(offset, offset+blocklen, self.datacache)
    # Copy the interesting bits to dest
    posinbytes = (pos % blocklen) * atomsize
    memcpy(dest, self.datacache + posinbytes, atomsize)
    # Update the cache index
    self.idxcache = idxcache
    return 1


  def __getitem__(self, object key):
    """
    x.__getitem__(key) <==> x[key]

    Returns values based on `key`.  All the functionality of
    ``ndarray.__getitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be returned as a NumPy
        array.

    See Also
    --------
    eval

    """

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
        if self.itemsize == self.atomsize:
          return PyArray_GETITEM(arr1, arr1.data)
        else:
          return arr1[0]
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
          count = key.sum()
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
    """
    x.__setitem__(key, value) <==> x[key] = value

    Sets values based on `key`.  All the functionality of
    ``ndarray.__setitem__()`` is supported (including fancy indexing), plus a
    special support for expressions:

    Parameters
    ----------
    key : string
        It will be interpret as a boolean expression (computed via `eval`) and
        the elements where these values are true will be set to `value`.

    See Also
    --------
    eval

    """

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
        chunk_ = chunk(cdata, self._dtype, self._cparams)
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
    nrows = self._nbytes // <npy_intp>self.atomsize
    if (start + blen) > nrows:
      blen = nrows - start

    # Fill `out` from data in chunks
    nwrow = 0
    stop = start + blen
    nchunks = self._nbytes // <npy_intp>self._chunksize
    chunklen = self._chunksize // self.atomsize
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
        chunk_._getitem(startb, stopb, out.data+nwrow*self.atomsize)
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

    vlen = boolarr.sum()   # number of true values in bool array
    value = utils.to_ndarray(value, self.dtype, arrlen=vlen)

    # Fill it from data in chunks
    nwrow = 0
    chunklen = self._chunklen
    nchunks = self._nbytes // <npy_intp>self._chunksize
    if self.leftover > 0:
      nchunks += 1
    nrows = self._nbytes // <npy_intp>self.atomsize
    for nchunk from 0 <= nchunk < nchunks:
      # Compute start & stop for each block
      startb, stopb, _ = clip_chunk(nchunk, chunklen, 0, nrows, 1)
      # Get boolean values for this chunk
      n = nchunk * chunklen
      boolb = boolarr[n+startb:n+stopb]
      blen = boolb.sum()
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
        chunk_ = chunk(cdata, self._dtype, self._cparams)
        self.chunks[nchunk] = chunk_
        # Update cbytes counter
        self._cbytes += chunk_.cbytes
      nwrow += blen

    # Safety check
    assert (nwrow == vlen)


  def __iter__(self):

    if not self.sss_mode:
      self.start = 0
      self.stop = self._nbytes // <npy_intp>self.atomsize
      self.step = 1
    if not (self.sss_mode or self.where_mode or self.wheretrue_mode):
      self.nhits = 0
      self.limit = sys.maxint
    # Initialize some internal values
    self.startb = 0
    self.nrowsread = self.start
    self._nrow = self.start - self.step
    self._row = -1  # a sentinel
    if self.where_mode and isinstance(self.where_arr, carray):
      self.nrowsinbuf = self.where_arr.chunklen
    else:
      self.nrowsinbuf = self._chunklen

    return self


  def iter(self, start=0, stop=None, step=1, limit=None):
    """
    iter(start=0, stop=None, step=1, limit=None)

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
    limit : int
        A maximum number of elements to return.  The default is return
        everything.

    Returns
    -------
    out : iterator

    See Also
    --------
    where, wheretrue

    """
    # Check limits
    if step <= 0:
      raise NotImplementedError, "step param can only be positive"
    self.start, self.stop, self.step = \
        slice(start, stop, step).indices(self.len)
    self.reset_sentinels()
    self.sss_mode = True
    if limit is not None:
      self.limit = limit
    return iter(self)


  def wheretrue(self, limit=None):
    """
    wheretrue(limit=None)

    Iterator that returns indices where this object is true.  Only useful for
    boolean carrays.

    Parameters
    ----------
    limit : int
        A maximum number of elements to return.  The default is return
        everything.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, where

    """
    # Check self
    if self.dtype.type != np.bool_:
      raise ValueError, "`self` is not an array of booleans"
    self.reset_sentinels()
    self.wheretrue_mode = True
    if limit is not None:
      self.limit = limit
    return iter(self)


  def where(self, boolarr, limit=None):
    """
    where(boolarr, limit=None)

    Iterator that returns values of this object where `boolarr` is true.

    Parameters
    ----------
    boolarr : a carray or NumPy array of boolean type
    limit : int
        A maximum number of elements to return.  The default is return
        everything.

    Returns
    -------
    out : iterator

    See Also
    --------
    iter, wheretrue

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
    if limit is not None:
      self.limit = limit
    return iter(self)


  def __next__(self):
    cdef char *vbool

    self.nextelement = self._nrow + self.step
    while (self.nextelement < self.stop) and (self.nhits < self.limit):
      if self.nextelement >= self.nrowsread:
        # Skip until there is interesting information
        while self.nextelement >= self.nrowsread + self.nrowsinbuf:
          self.nrowsread += self.nrowsinbuf
        # Compute the end for this iteration
        self.stopb = self.stop - self.nrowsread
        if self.stopb > self.nrowsinbuf:
          self.stopb = self.nrowsinbuf
        self._row = self.startb - self.step

        # Skip chunks with zeros if in wheretrue_mode
        if self.wheretrue_mode and self.check_zeros(self):
          self.nrowsread += self.nrowsinbuf
          self.nextelement += self.nrowsinbuf
          continue

        if self.where_mode:
          # Skip chunks with zeros in where_arr
          if self.check_zeros(self.where_arr):
            self.nrowsread += self.nrowsinbuf
            self.nextelement += self.nrowsinbuf
            continue
          # Read a chunk of the boolean array
          self.where_buf = self.where_arr[
            self.nrowsread:self.nrowsread+self.nrowsinbuf]

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
          self.nhits += 1
          return self._nrow
        else:
          continue
      if self.where_mode:
        vbool = <char *>(self.where_buf.data + self._row)
        if not vbool[0]:
            continue
      self.nhits += 1
      # Return the current value in I/O buffer
      if self.itemsize == self.atomsize:
        return PyArray_GETITEM(
          self.iobuf, self.iobuf.data + self._row * self.atomsize)
      else:
        return self.iobuf[self._row]

    else:
      # Release buffers
      self.iobuf = np.empty(0, dtype=self.dtype)
      self.where_buf = np.empty(0, dtype=np.bool_)
      self.reset_sentinels()
      raise StopIteration        # end of iteration


  cdef reset_sentinels(self):
    """Reset sentinels for iterator."""
    self.sss_mode = False
    self.wheretrue_mode = False
    self.where_mode = False
    self.where_arr = None
    self.nhits = 0
    self.limit = sys.maxint


  cdef int check_zeros(self, object barr):
    """Check for zeros.  Return 1 if all zeros, else return 0."""
    cdef int bsize
    cdef npy_intp nchunk
    cdef carray carr
    cdef ndarray ndarr
    cdef chunk chunk_

    if isinstance(barr, carray):
      # Check for zero'ed chunks in carrays
      carr = barr
      nchunk = self.nrowsread // <npy_intp>self.nrowsinbuf
      if nchunk < len(carr.chunks):
        chunk_ = carr.chunks[nchunk]
        if chunk_.isconstant and chunk_.constant in (0, ''):
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
