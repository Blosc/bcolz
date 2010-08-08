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

    __version__
"""

import sys
import numpy


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
    BLOSC_VERSION_STRING, BLOSC_VERSION_DATE

  void blosc_get_versions(char *version_str, char *version_date)
  int blosc_set_nthreads(int nthreads)
  unsigned int blosc_compress(int clevel, int doshuffle, size_t typesize,
                              size_t nbytes, void *src, void *dest,
                              size_t destsize)
  unsigned int blosc_decompress(void *src, void *dest, size_t destsize)
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

  - none yet
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
  cdef npy_intp nbytes, cbytes
  cdef void *data

  def __cinit__(self, ndarray array, int clevel=5, int shuffle=1):
    """Initialize and compress data based on passed `array`.

    You can pass `clevel` and `shuffle` params to the internal compressor.
    """
    cdef int i, itemsize
    cdef npy_intp nbytes, cbytes

    self.dtype = dtype = array.dtype
    self.shape = shape = array.shape
    itemsize = dtype.itemsize
    nbytes = itemsize
    for i in self.shape:
      nbytes *= i
    self.data = malloc(nbytes)
    # Compress up to nbytes-1 maximum
    cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, array.data,
                            self.data, nbytes-1)
    if cbytes < 0:
      raise RuntimeError, "Fatal error during Blosc compression: %d" % cbytes
    if cbytes == 0:
      # Compression not successful.  Copy data directly.
      memcpy(self.data, array.data, nbytes)
      cbytes = nbytes
    # Set size info for the instance
    self.cbytes = cbytes
    self.nbytes = nbytes


  def toarray(self):
    """Convert this `carray` instance into a NumPy array."""
    cdef ndarray array
    cdef int ret

    # Build a NumPy container
    array = numpy.empty(shape=self.shape, dtype=self.dtype)
    # Fill it with uncompressed data
    if self.cbytes == self.nbytes:
      memcpy(array.data, self.data, self.nbytes)
    else:
      ret = blosc_decompress(self.data, array.data, self.nbytes)
      if ret <= 0:
        raise RuntimeError, "Fatal error during Blosc decompression: %d" % ret
    return array


  def __getitem__(self, key):
    """__getitem__(self, key) -> values
    """
    raise NotImplementedError


  def __setitem__(self, object key, object value):
    """__setitem__(self, key, value) -> None
    """
    raise NotImplementedError


  def __str__(self):
    """Represent the carray as an string."""
    return str(self.toarray())


  def __repr__(self):
    """Represent the record as an string."""
    cratio = self.nbytes / float(self.cbytes)
    fullrepr = "nbytes: %d; cbytes: %d; compr. ratio: %.2f\n%r" % \
        (self.nbytes, self.cbytes, cratio, self.toarray())
    return fullrepr


  def __dealloc__(self):
    """Release C resources before destruction."""
    free(self.data)



## Local Variables:
## mode: python
## py-indent-offset: 2
## tab-width: 2
## fill-column: 78
## End:
