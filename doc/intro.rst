------------
Introduction
------------

carray at glance
================

carray is a Python package that provides containers (called `carray`
and `ctable`) for numerical data that can be compressed in-memory.  It
is highly based on NumPy, and uses it as the standard data container
to communicate with carray objects.

The building blocks of carray objects are the so-called ``chunks``
that are bits of data compressed as a whole, but that can be
decompressed partially in order to improve the fetching of small parts
of the array.  This ``chunked`` nature of the carray objects, together
with a buffered I/O, makes appends very cheap and fetches reasonably
fast (although modification of values is an expensive operation).

The compression/decompression process is carried out internally by
Blosc, a high-performance compressor that is optimized for binary
data.  That ensures maximum performance for I/O operation.

carray and ctable objects
-------------------------

The main objects in the carray package are:

  * `carray`: container for homogeneous data
  * `ctable`: container for heterogeneous data

`carray` is very similar to a NumPy `ndarray` in that it supports the
same types and data access interface.  The main difference between
them is that a `carray` can keep data compressed in-memory, allowing
to deal with larger datasets with the same amount of RAM.  And another
important difference is the chunked nature of the `carray` that allows
data to be appended much more efficiently.

On his hand, a `ctable` is also similar to a NumPy ``structured
array``, that shares the same properties with its `carray` brother,
namely, compression and chunking.  In addition, data is stored in a
column-wise order and not on a row-wise order, as the ``structured
array``, allowing for very cheap column handling.  This is of
paramount importance when you need to add and remove columns in wide
(and possibly large) in-memory tables --doing this with regular
``structured arrays`` in NumPy is exceedingly slow.

Also, column-wise ordering turns out that this gives the `ctable` a
huge opportunity to improve compression ratio.  This is because data
tends to expose more similarity in elements that are contiguous in
columns rather than those in the same row, so compressors generally do
a much better job when data is aligned column-wise.  This is specially
true with Blosc because its special ``shuffle`` filter does a much
better job with homogeneous data (columns) than with heterogeneous data
(rows).

carray main features
--------------------

carray objects bring several advantages over plain NumPy objects:

  * Data is compressed: they take less memory space.

  * Efficient appends: you can append more data at the end of the
    objects very quickly.

  * `ctable` objects have the data arranged column-wise.  This allows
    for much better performance when working with big tables, as well
    as for improving the compression ratio.

  * Numexpr-powered: you can operate with compressed data in a fast
    and convenient way.  Blosc ensures that the additional overhead of
    handling compressed data natively is very low (the secret goal is
    to make compressed data operations to perform actually faster :-).


carray limitations
------------------

carray also comes with drawbacks:

  * Lack of support for the complete NumPy functionality.  Although
    Numexpr will let you to perform a wide-range of operations on
    native carray/ctable objects, NumPy exposes a much richer toolset.

  * At this time, carray objects can only be uni-dimensional (newer
    versions might get rid of this limitation though).

