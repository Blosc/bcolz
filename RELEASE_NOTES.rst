=======================
Release notes for bcolz
=======================

Changes from 0.8.0 to 0.8.1
===========================

- Downgrade to Blosc v1.4.1 (#144 @esc)

- Fix license include (#143 @esc)

- Upgrade to Cython 0.22 (#145 @esc)

Changes from 0.7.3 to 0.8.0
===========================

- Public API for ``carray`` (#98 @FrancescElies and #esc)

  A Cython definition file ``carrat_ext.pxd`` was added that contains the
  definitions for the ``carray``, ``chunks`` and ``chunk`` classes. This was
  done to allow more complex programs to be built on the compressed container
  primitives provided by bcolz.

- Overhaul the release procedure

- Other miscellaneous fixes and improvements

Changes from 0.7.2 to 0.7.3
===========================

- Update to Blosc ``v1.5.2``

- Added support for pickling persistent carray/ctable objects.  Basically,
  what is serialized is the ``rootdir`` so the data is still sitting on disk
  and the original contents in ``rootdir`` are still needed for unpickling.
  (#79 @mrocklin)

- Fixed repr-ing of ``datetime64`` ``carray`` objects (#99 @cpcloud)

- Fixed Unicode handling for column addressing (#91 @CarstVaartjes)

- Conda recipe and ``.binstar.yml`` (#88 @mrocklin and @cpcloud)

- Removed ``unittest2`` as a run-time dependency (#90 @mrocklin)

- Various typo fixes. (#75 @talumbau, #86 @catawbasam and #83 @bgrant)

- Other miscellaneous fixes and improvements


Changes from 0.7.1 to 0.7.2
===========================

- Fix various test that were failing on 32 bit, especially a segfault

- Fix compatibility with Numpy 1.9

- Fix compatibility with Cython 0.21.

- Allow tests to be executed with ``nosetests``.

- Include git hash in version info when applicable.

- Initial support for testing on Travis CI.

- Close file handle when ``nodepath`` arg to ``ctable.fromhdf5`` is incorrect.

- Introduced a new ``carray.view()`` method returning a light-weight
  carray object describing the same data than the original carray.  This
  is mostly useful for iterators, but other uses could be devised as
  well.

- Each iterator now return a view (see above) of the original object, so
  things like::

      >>> bc = bcolz.ctable([[1, 2, 3], [10, 20, 30]], names=['a', 'b'])
      >>> bc.where('a >= 2')  # call .where but don't do anything with it
      <itertools.imap at 0x7fd7a84f5750>
      >>> list(bc['b'])  # later iterate over table, get where result
      [10, 20, 30]

  works as expected now.

- Added a workaround for dealing with Unicode types.

- Fix writing absolute paths into the persistent metadata.

- ``next(carray)`` calls now work as they should.

- Fix the ``__repr__`` method of the ``chunk`` class.

- Prevent sometimes incorrect assignment of dtype to name with fromhdf5.

- Various miscellaneous bug-fixes, pep8 improvements and typo-fixes.


Changes from 0.7.0 to 0.7.1
===========================

- Return the outcome of the test for checking that in standalone
  programs.  Thanks to Ilan Schnell for suggesting that.

- Avoiding importing lists of ints as this has roundtrip problems in
  32-bit platforms.

- Got rid of the nose dependency for Python 2.6.  Thanks to Ilan Schnell
  for the suggestion.


Changes from 0.5.1 to 0.7.0
===========================

- Renamed the ``carray`` package to ``bcolz``.

- Added support for Python 3.

- Added a new function `iterblocks` for quickly returning blocks of
  data, not just single elements. ctable receives a new `whereblocks`
  method, which is the equivalent of `where` but returning data blocks.

- New pandas import/export functionality via `ctable.fromdataframe()`
  and `ctable.todataframe()`.

- New HDF5/PyTables import/export functionality via `ctable.fromhdf5()`
  and `ctable.tohdf5()`.

- Support for c-blosc 1.4.1.  This allows the use of different
  compressors via the new `cname` parameter in the `cparams` class, and
  also to be used in platforms not supporting unaligned access.

- Objects are supported in carray containers (not yet for ctable).

- Added a new `free_cachemem()` method for freeing internal caches after
  reading/querying carray/ctable objects.

- New `cparams.setdefaults()` method for globally setting defaults in
  compression parameters during carray/ctable creation.

- Disabled multi-threading in both Blosc and numexpr because it is not
  delivering the promised speedups yet.  This can always be re-activated
  by using `blosc_set_nthreads(nthreads)` and
  `numexpr.set_num_threads(nthreads)`.


Changes from 0.5 to 0.5.1
=========================

- Added the missing bcolz.tests module in setup.py.


Changes from 0.4 to 0.5
=======================

- Introduced support for persistent objects.  Now, every carray and
  ctable constructor support a new `rootdir` parameter where you can
  specify the path where you want to make the data stored.

  The format chosen is explained in the 'persistence.rst' file, except
  that the blockpack format is still version 1 (that will probably
  change in future versions).  Also, JSON is used for storing metadata
  instead of YAML.  This is mainly for avoiding a new library
  dependency.

- New `open(rootdir, mode='a')` top level function so as to open on-disk
  bcolz objects.

- New `flush()` method for `carray` and `ctable` objects.  This is
  useful for flushing data to disk in persistent objects.

- New `walk(dir, classname=None, mode='a')` top level function for
  listing carray/ctable objects handing from `dir`.

- New `attrs` accessor is provided, so that users can store
  its own metadata (in a persistent way, if desired).

- Representation of carray/ctable objects is based now on the same code
  than NumPy.

- Reductions (`sum` and `prod`) work now, even with the `axis` parameter
  (when using the Numexpr virtual machine).


Changes from 0.3.2 to 0.4
=========================

- Implemented a `skip` parameter for iterators in `carray` and `ctable`
  objects.  This complements `limit` for selecting the number of
  elements to be returned by the iterator.

- Implemented multidimensional indexing for carrays.  Than means that
  you can do::

    >>> a = ca.zeros((2,3))

  Now, you can access any element in any dimension::

    >>> a[1]
    array([ 0.,  0.,  0.])
    >>> a[1,::2]
    array([ 0., 0.])
    >>> a[1,1]
    0.0

- `dtype` and `shape` attributes follow now ndarray (NumPy) convention.
  The `dtype` is always a scalar and the dimensionality is added to the
  `shape` attribute.  Before, all the additional dimensionality was in
  the `dtype`.  The new convention should be more familiar for
  everybody.


Changes from 0.3.1 to 0.3.2
===========================

- New `vm` parameter for `eval()` that allows to choose a 'python' or
  'numexpr' virtual machine during operations.  If numexpr is not
  detected, the default will be 'python'.

  That means that you can use any function available in Python for
  evaluating bcolz expressions and that numexpr is not necessary
  anymore for using `eval()`.

- New `out_flavor` parameter for `eval()` that allows to choose the
  output type.  It can be 'bcolz' or 'numpy'.

- New `defaults.py` module that enables the user to modify the defaults
  for internal bcolz operation.  Defaults that are currently
  implemented: `eval_out_flavor` and `eval_vm`.

- Fixed a bug with `carray.sum()` for multidimensional types.


Changes from 0.3 to 0.3.1
=========================

- Added a `limit` parameter to `iter`, `where` and `wheretrue` iterators
  of carray object and to `iter` and `where` of ctable object.

- Full support for multidimensional carrays.  All types are supported,
  except the 'object' type (that applies to unidimensional carrays too).

- Added a new `reshape()` for reshaping to new (multidimensional)
  carrays.  This supports the same functionality than `reshape()` in
  NumPy.

- The behaviour of a carray was altered after using an iterator.  This
  has been fixed.  Thanks to Han Genuit for reporting.


Changes from 0.2 to 0.3
=======================

- Added a new `ctable` class that implements a compressed, column-wise
  table.

- New `arange()` constructor for quickly building carray objects (this
  method is much faster than using `fromiter()`).

- New `zeros()` constructor for quickly building zeroed carray objects.
  This is way faster than its NumPy counterpart.

- New `ones()` constructor for quickly building 1's carray objects.
  Very fast.

- New `fill()` constructor for quickly building carray objects with a
  filling value.  This is very fast too.

- New `trim()` method for `carray` and `ctable` objects for trimming
  items.

- New `resize()` method for `carray` and `ctable` objects for resizing
  lengths.

- New `test()` function that runs the complete test suite.

- Added a new `eval()` function to evaluate expressions including any
  combination of carrays, ndarrays, sequences or scalars.  Requires
  Numexpr being installed.

- Added new `__len__()` and `__sizeof__()` special methods for both
  `carray` and `ctable` objects.

- New `sum()` method for `carray` that computes the sum of the array
  elements.

- Added new `nbytes` and `cbytes` properties for `carray` and `ctable`
  objects.  The former accounts for the size of the original
  (non-compressed) object, and the later for the actual compressed
  object.

- New algorithm for computing an optimal chunk size for carrays based on
  the new `expectedlen` argument.

- Added `chunklen` property for `carray` that allows querying the chunk
  length (in rows) for the internal I/O buffer.

- Added a new `append(rows)` method to `ctable` class.

- Added a new `wheretrue()` iterator for `carray` that returns the
  indices for true values (only valid for boolean arrays).

- Added a new `where(boolarr)` iterator for `carray` that returns the
  values where `boolarr` is true.

- New idiom ``carray[boolarr]`` that returns the values where `boolarr`
  is true.

- New idiom ``ctable[boolarr]`` that returns the rows where `boolarr` is
  true.

- Added a new `eval()` method for `ctable` that is able to evaluate
  expressions with columns.  It needs numexpr to be installed.

- New idiom ``ctable[boolexpr]`` that returns the rows fulfilling the
  boolean expression.  Needs numexpr.

- Added fancy indexing (as a list of integers) support to `carray` and
  `ctable`.

- Added `copy(clevel, shuffle)` method to both `carray` and `ctable`
  objects.

- Removed the `toarray()` method in `carray` as this was equivalent to
  ``carray[:]`` idiom.

- Renamed `setBloscMaxThreads()` to `blosc_set_num_threads()` and
  `whichLibVersion()` to `blosc_version()` to follow bcolz name
  conventions more closely.

- Added a new `set_num_threads()` to set the number of threads in both
  Blosc and Numexpr (if available).

- New `fromiter()` constructor for creating `carray` objects from
  iterators.  It follows the NumPy API convention.

- New `cparams(clevel=5, shuffle=True)` class to host all params related
  with compression.

- Added more indexing support for `carray.__getitem__()`.  All indexing
  modes present in NumPy are supported now, including fancy indexing.
  The only exception are negative steps in ``carray[start:stop:-step]``.

- Added support for `bcolz.__setitem__()`.  All indexing modes present
  in NumPy are supported, including fancy indexing.  The only exception
  are negative steps in ``carray[start:stop:-step] = values``.

- Added support for `ctable.__setitem__()`.  All indexing modes present
  in NumPy are supported, including fancy indexing.  The only exception
  are negative steps in ``ctable[start:stop:-step] = values``.

- Added new `ctable.__iter__()`, `ctable.iter()` and `ctable.where()`
  iterators mimicking the functionality in carray object.


Changes from 0.1 to 0.2
=======================

- Added a couple of iterators for carray: `__iter__()` and `iter(start,
  stop, step)`.  The difference is that the later does accept slices.

- Added a `__len__()` method.


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
