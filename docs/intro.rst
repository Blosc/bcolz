------------
Introduction
------------

bcolz at glance
===============

bcolz provides columnar, chunked data containers that can be
compressed either in-memory and on-disk.  Column storage allows for
efficiently querying tables, as well as for cheap column addition and
removal.  It is based on `NumPy <http://www.numpy.org>`_, and uses it
as the standard data container to communicate with bcolz objects, but
it also comes with support for import/export facilities to/from
`HDF5/PyTables tables <http://www.pytables.org>`_ and `pandas
dataframes <http://pandas.pydata.org>`_.

The building blocks of bcolz objects are the so-called ``chunks`` that
are bits of data compressed as a whole, but that can be (partially)
decompressed in order to improve the fetching of small parts of the
array.  This ``chunked`` nature of the bcolz objects, together with a
buffered I/O, makes appends very cheap and fetches reasonably fast
(although the modification of values can be an expensive operation).

The compression/decompression process is carried out internally by
Blosc, a high-performance compressor that is optimized for binary
data.  The fact that Blosc splits chunks internally in so-called
blocks means that only the interesting part of the chunk will
decompressed (typically in L1 or L2 caches). That ensures maximum
performance for I/O operation (`either on-disk or in memory
<https://github.com/FrancescAlted/DataContainersTutorials>`_).

bcolz can use numexpr or dask internally (numexpr is used by default
if installed, then dask and if these are not found, then the pure
Python interpreter) so as to accelerate many internal vector and query
operations (although it can use pure NumPy for doing so too).  numexpr
can optimize memory (cache) usage and uses multithreading for doing
the computations, so it is blazing fast.  This, in combination with
carray/ctable disk-based, compressed containers, can be used for
performing out-of-core computations efficiently, but most importantly
*transparently*.


carray and ctable objects
-------------------------

The main data container objects in the bcolz package are:

  * `carray`: container for homogeneous & heterogeneous (row-wise) data
  * `ctable`: container for heterogeneous (column-wise) data

`carray` is very similar to a NumPy `ndarray` in that it supports the
same types and basic data access interface.  The main difference
between them is that a `carray` can keep data compressed (both
in-memory and on-disk), allowing to deal with larger datasets with the
same amount of memory/disk.  And another important difference is the
chunked nature of the `carray` that allows data to be appended much
more efficiently.

On his hand, a `ctable` is also similar to a NumPy ``structured
array`` that shares the same properties with its `carray` brother,
namely, compression and chunking.  Another difference is that data is
stored in a column-wise order (and not on a row-wise, like the
``structured array``), allowing for very cheap column handling.  This
is of paramount importance when you need to add and remove columns in
wide (and possibly large) in-memory and on-disk tables --doing this
with regular ``structured arrays`` in NumPy is exceedingly slow.

Furthermore, columnar means that the tabular datasets are stored
column-wise order, and this turns out to offer better opportunities to
improve compression ratio.  This is because data tends to expose more
similarity in elements that sit in the same column rather than those
in the same row, so compressors generally do a much better job when
data is aligned in such column-wise order.


bcolz main features
--------------------

bcolz objects bring several advantages over plain NumPy objects:

  * Data is compressed: they take less storage space.

  * Efficient shrinks and appends: you can shrink or append more data
    at the end of the objects very efficiently (i.e. copies of the
    whole array are not needed).

  * Persistence comes seamlessly integrated, so you can work with
    on-disk arrays almost in the same way than with in-memory ones
    (bar some special attention to flush data being required).

  * `ctable` objects have the data arranged column-wise.  This allows
    for much better performance when working with big tables, as well
    as for improving the compression ratio.

  * Can leverage Numexpr and Dask as virtual machines for fast
    operation with bcolz objects.  Blosc ensures that the additional
    overhead of handling compressed data natively is very low.

  * Advanced query capabilities.  The ability of a `ctable` object to
    iterate over the rows whose fields fulfill some conditions (and
    evaluated via numexpr, dask or pure python virtual machine) allows
    to perform queries very efficiently.


bcolz limitations
------------------

bcolz does not currently come with good support in the next areas:

  * Limited number of operations, at least when compared with NumPy.
    The supported operations are basically vectorized ones (i.e. those
    that are made element-by-element).  But with is changing with the
    adoption of additional kernels like `Dask
    <https://github.com/dask/dask>`_ (and more to come).

  * Limited broadcast support.  For example, NumPy lets you operate
    seamlessly with arrays of different shape (as long as they are
    compatible), but you cannot do that with bcolz.  The only object
    that can be broadcasted currently are scalars
    (e.g. ``bcolz.eval("x+3")``).

  * Some methods (namely `carray.where()` and `carray.wheretrue()`)
    do not have support for multidimensional arrays.

  * Multidimensional `ctable` objects are not supported.  However, as
    the columns of these objects can be fully multidimensional, this
    is not regarded as an important limitation.
