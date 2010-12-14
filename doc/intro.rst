------------
Introduction
------------

carray at glance
================

carray is a Python package that provides containers for numerical data
that can be compressed in-memory.  It is highly based on NumPy, and
uses it as the standard data container to communicate with carray
objects.

The building blocks of carray objects are the so-called ``chunks``
that are compressed as a whole, but they can be decompressed partially
in order to improve the fetching of small parts of the array.  This
``chunked`` nature of the carray objects, together with a buffer for
performing I/O, makes appends very cheap and fetches reasonably fast
(however, modification of values is an expensive operation).

The compression/decompression process is carried out internally by
Blosc, a high-performance compressor that is optimized for binary
data.  That ensures maximum performance for I/O operation.

carray objects bring several advantages over NumPy::

  * Data is compressed: they take less memory space.

  * Efficient appends: you can append more data at the end of the
    objects very quickly.

  * Numexpr-powered: you can operate with compressed data in a
    fast and convenient way.

  * Tables (`ctable`) have the data arranged column-wise.  This allows
    for much better performance when working with wide tables, as well
    as for adding and deleting columns.

carray limitations
==================

At this time, carray objects can only be uni-dimensional.  Newer
versions of the carray package might get rid of this limitation
though.

