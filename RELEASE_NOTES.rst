=======================
Release notes for bcolz
=======================


Changes from 1.2.1 to 1.2.2
===========================

  #XXX version-specific blurb XXX#


Changes from 1.2.0 to 1.2.1
===========================

- C-Blosc internal sources updated to 1.14.3.  This basically means that
  internal Zstd sources are bumped to 1.3.4, which may lead to noticeable
  improved speeds (specially for low compression ratios).

- `np.datetime64` and other scalar objects that have `__getitem__()` are now
  supported in _eval_blocks().  PR #377.  Thanks to apalepu23.

- Vendored cpuinfo.py updated to 4.0.0 (ARM aarch64 is recognized now).

- Allow setup.py to work even if not on Intel or ARM or PPC archs are found.


Changes from 1.1.2 to 1.2.0
===========================

- Support for Python <= 2.6 or Python <= 3.4 has been deprecated.

- C-Blosc internal sources updated to 1.14.2.  Using a C-Blosc library
  > 1.14 is important for forward compatibility.  For more info see:
  http://blosc.org/posts/new-forward-compat-policy/


Changes from 1.1.1 to 1.1.2
===========================

- Updated setup.py to include Zstd codec in Blosc.  Fixes #331.


Changes from 1.1.0 to 1.1.1
===========================

- Allow to delete all the columns in a ctable.  Fixes #306.

- Double-check the value of a column that is being overwritten.  Fixes
  #307.

- Use `pkg_resources.parse_version()` to test for version of packages.
  Fixes #322.

- Now all the columns in a ctable are enforced to be a carray instance
  in order to simplify the internal logic for handling columns.

- Now, the cparams are preserved during column replacement, e.g.:

  `ct['f0'] = x + 1`

  will continue to use the same cparams than the original column.

- C-Blosc updated to 1.11.2.

- Added a new `defaults_ctx` context so that users can select defaults
  easily without changing global behaviour. For example::

   with bcolz.defaults_ctx(vm="python", cparams=bcolz.cparams(clevel=0)):
      cout = bcolz.eval("(x + 1) < 0")

- Fixed a crash occurring in `ctable.todataframe()` when both `columns`
  and `orient='columns'` were specified.  PR #311.  Thanks to Peter
  Quackenbush.


Changes from 1.0.0 to 1.1.0
===========================

- Defaults when creating carray/ctable objects are always scalars now.
  The new approach follows what was documented and besides it prevents
  storing too much JSON data in meta/ directory.

- Fixed an issue with bcolz.iterblocks() not working on multidimensional
  carrays.

- It is possible now to create ctables with more than 255 columns.  Thanks
  to Skipper Seabold.  Fixes #131 (via PR #303).

- Added a new `quantize` filter for allowing lossy compression on
  floating point data.  Data is quantized using
  np.around(scale*data)/scale, where scale is 2**bits, and bits is
  determined from the quantize value.  For example, if quantize=1, bits
  will be 4.  0 means that the quantization is disabled.

  Here it is an example of what you can get from the new quantize::

    In [9]: a = np.cumsum(np.random.random_sample(1000*1000)-0.5)

    In [10]: bcolz.carray(a, cparams=bcolz.cparams(quantize=0))  # no quantize
    Out[10]:
    carray((1000000,), float64)
      nbytes: 7.63 MB; cbytes: 6.05 MB; ratio: 1.26
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=0)
    [ -2.80946077e-01  -7.63925274e-01  -5.65575047e-01 ...,   3.59036158e+02
       3.58546624e+02   3.58258860e+02]

    In [11]: bcolz.carray(a, cparams=bcolz.cparams(quantize=1))
    Out[11]:
    carray((1000000,), float64)
      nbytes: 7.63 MB; cbytes: 1.41 MB; ratio: 5.40
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=1)
    [ -2.50000000e-01  -7.50000000e-01  -5.62500000e-01 ...,   3.59036158e+02
       3.58546624e+02   3.58258860e+02]

    In [12]: bcolz.carray(a, cparams=bcolz.cparams(quantize=2))
    Out[12]:
    carray((1000000,), float64)
      nbytes: 7.63 MB; cbytes: 2.20 MB; ratio: 3.47
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=2)
    [ -2.81250000e-01  -7.65625000e-01  -5.62500000e-01 ...,   3.59036158e+02
       3.58546624e+02   3.58258860e+02]

    In [13]: bcolz.carray(a, cparams=bcolz.cparams(quantize=3))
    Out[13]:
    carray((1000000,), float64)
      nbytes: 7.63 MB; cbytes: 2.30 MB; ratio: 3.31
      cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=3)
    [ -2.81250000e-01  -7.63671875e-01  -5.65429688e-01 ...,   3.59036158e+02
       3.58546624e+02   3.58258860e+02]

  As you can see, the compression ratio can improve pretty significantly
  when using the quantize filter.  It is important to note that by using
  quantize you are loosing precision on your floating point data.

  Also note how the first elements in the quantized arrays have less
  significant digits, but not the last ones.  This is a side effect due
  to how bcolz stores the trainling data that do not fit in a whole
  chunk.  But in general you should expect a loss in precision.

- Fixed a bug in carray.__getitem__() when the chunksize was not an
  exact multiple of the blocksize.  Added test:
  test_carray.py::getitemMemoryTest::test06.

- bcolz now follows the convention introduced in NumPy 1.11 for
  representing datetime types with TZ="naive" (i.e. with no TZ info in
  the representation).  See https://github.com/numpy/numpy/blob/master/doc/release/1.11.0-notes.rst#datetime64-changes.

- bcolz now releases the GIL during Blosc compression/decompression.  In
  multi-threaded environments, a single-threaded, contextual version of
  Blosc is used instead (this is useful for frameworks like Dask).

- Removed from the ``cbytes`` count the storage overhead due to the
  internal container.  This overhead was media-dependent, and it was
  just a guess anyway.

- The -O1 compilation flag has been removed and bcolz is compiled now at
  full optimization.  I have tested that for several weeks, without any
  segfault, so this should be pretty safe.

- Added information about the chunklen, chunksize and blocksize (the
  size of the internal blocks in a Blosc chunk) in the repr() of a
  carray.

- New accelerated codepath for `carray[:] = array` assignation.  This
  operation should be close in performance to `carray.copy()` now.

- carray object does implement the __array__() special method
  (http://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.classes.html#numpy.class.__array__)
  now. With this, interoperability with numpy arrays is easier and
  faster:

  Before __array__()::
    >>> a = np.arange(1e7)
    >>> b = np.arange(1e7)
    >>> ca = bcolz.carray(a)
    >>> cb = bcolz.carray(b)
    >>> %timeit ca + a
    1 loop, best of 3: 1.06 s per loop
    >>> %timeit np.array(bcolz.eval("ca*(cb+1)"))
    1 loop, best of 3: 1.18 s per loop

  After __array__()::
    >>> %timeit ca + a
    10 loops, best of 3: 45.2 ms per loop
    >>> %timeit np.array(bcolz.eval("ca*(cb+1)"))
    1 loop, best of 3: 133 ms per loop

  And it also allows to use bcolz carrays more efficiently in some scenarios::
    >>> import numexpr
    >>> %timeit numexpr.evaluate("ca*(cb+1)")
    10 loops, best of 3: 76.2 ms per loop
    >>> %timeit numexpr.evaluate("a*(b+1)")
    10 loops, best of 3: 25.5 ms per loop  # ndarrays are still faster

- Internal C-Blosc sources bumped to 1.9.2.

- Dask (dask.pydata.org) is supported as another virtual machine backed
  for bcolz.eval().  Now, either Numexpr (the default) or Dask or even
  the Python interpreter can be used to evaluate complex expressions.

- The default compressor has been changed from 'blosclz' to 'lz4'.
  BloscLZ tends to be a bit faster when decompressing, but LZ4 is
  quickly catching up as the compilers are making progress with memory
  access optimizations.  Also, LZ4 is considerably faster during
  compression and in general compresses better too.

- The supported SIMD extensions (SSE2 and AVX2) of the current platform
  are auto-detected so that the affected code will selectively be
  included from vendored C-Blosc sources.

- Added a new `blen` parameter to bcolz.eval() so that the user can
  select the length of the operand blocks to be operated with.

- New fine-tuning of the automatically computed blen in bcolz.eval() for
  better times and reduced memory consumption.

- Added a new `out_flavor` parameter to the ctable.iter() and
  ctable.where() for specifying the type of result rows.  Now one can
  select namedtuple (default), tuple or ndarray.

- The performance of carray.whereblocks() has been accelerated 2x due to
  the internal use of tuples instead of named tuples.

- New ctable.fetchwhere() method for getting the rows fulfilling some
  condition in one go.

- Parameter `outfields` in ctable.whereblocks has been renamed to
  `outcols` for consistency with the other methods.  The previous
  'outfields' name is considered a bug and hence is not supported
  anymore.

- bcolz.fromiter() has been streamlined and optimized.  The result is
  that it uses less memory and can go faster too (20% ~ 50%, depending
  on the use).

- The values for defaults.eval_out_flavor has been changed to ['bcolz',
  'numpy'] instead of previous ['carray', 'numpy'].  For backward
  compatibility the 'carray' value is still allowed.

- The `bcolz.defaults.eval_out_flavor` and `bcolz.defaults.eval_vm` have
  been renamed to `bcolz.defaults.out_flavor` and `bcolz.defaults.vm`
  because they can be used in other places than just bcolz.eval().  The
  old `eval_out_flavor` and `eval_vm` properties of the `defaults`
  object are still kept for backward compatibility, but they are not
  documented anymore and its use is discouraged.

- Added a new `user_dict` parameter in all ctable methods that evaluate
  expressions.  For convenience, this dictionary is updated internally
  with ctable columns, locals and globals from the caller.

- Small optimization for using the recently added re_evaluate() function
  in numexpr for faster operation of numexpr inside loops using the same
  expression (quite common scenario).

- Unicode strings are recognized now when imported from a pandas
  dataframe, making the storage much more efficient.  Before unicode was
  converted into 'O'bject type, but the change to 'U'nicode should be
  backward compatible.

- Added a new `vm` parameter to specify the virtual machine for doing
  internal operations in ctable.where(), ctable.fetchwhere() and
  ctable.whereblocks().


Changes from 0.12.1 to 1.0.0
============================

- New version of embedded C-Blosc (bumped to 1.8.1).  This allows for
  using recent C-Blosc features like the BITSHUFFLE filter that
  generally allows for better compression ratios at the expense of some
  slowdown.  Look into the carray tutorial on how to use the new
  BITSHUFFLE filter.

- Use the -O1 flag for compiling the included C-Blosc sources on Linux.
  This represents slower performance, but fixes nasty segfaults as can
  be seen in issue #110 of python-blosc.  Also, it prints a warning for
  using an external C-Blosc library.

- Improved support for operations with carrays of shape (N, 1). PR #296.
  Fixes #165 and #295.  Thanks to Kevin Murray.

- Check that column exists before inserting a new one in a ctable via
  `__setitem__`.  If it exists, the existing column is overwritten.
  Fixes #291.

- Some optimisations have been made within ``carray.__getitem__`` to
  improve performance when extracting a slice of data from a
  carray. This is particularly relevant when running some computation
  chunk-by-chunk over a large carray. (#283 @alimanfoo).


Changes from 0.12.0 to 0.12.1
=============================

- ``setup.py`` now defers operations requiring ``numpy`` and ``Cython``
  until after those modules have been installed by ``setuptools``.  This
  means that users no longer need to pre-install ``numpy`` and
  ``Cython`` to install ``bcolz``.


Changes from 0.11.4 to 0.12.0
=============================

- Fixes an installation glitch for Windows. (#268 @cgohlke).

- The tutorial is now a Jupyter notebook. (#261 @FrancescElies).

- Replaces numpy float string specifier in test with numpy.longdouble
  (#271 @msarahan).

- Fix for allowing the use of variables of type string in `eval()` and
  other queries. (#273, @FrancescAlted).

- The size of the tables during import/export to HDF5 are honored now
  via the `expectedlen` (bcolz) and `expectedrows` (PyTables)
  parameters (@FrancescAlted).

- Update only the valid part of the last chunk during boolean
  assignments.  Fixes a VisibleDeprecationWarning with NumPy 1.10
  (@FrancescAlted).

- More consistent string-type checking to allow use of unicode strings
  in Python 2 for queries, column selection, etc. (#274 @BrenBarn).

- Installation no longer fails when listed as dependency of project
  installed via setup.py develop or setup.py install. (#280 @mindw,
  fixes #277).

- Paver setup has been deprecated (see #275).


Changes from 0.11.3 to 0.11.4
=============================

- The .pyx extension is not packed using the absolute path anymore.
  (#266 @FrancescAlted)


Changes from 0.11.2 to 0.11.3
=============================

- Implement feature #255 bcolz.zeros can create new ctables too, either
  empty or filled with zeros. (#256 @FrancescElies @FrancescAlted)


Changes from 0.11.1 to 0.11.2
=============================

- Changed the `setuptools>18.3` dependency to `setuptools>18.0` because
  Anaconda does not have `setuptools > 18.1` yet.


Changes from 0.11.0 to 0.11.1
=============================

- Do not try to flush when a ctable is opened in 'r'ead-only mode.
  See issue #252.

- Added the mock dependency for Python2.

- Added a `setuptools>18.3` dependency.

- Several fixes in the tutorial (Francesc Elies).


Changes from 0.10.0 to 0.11.0
=============================

- Added support for appending a np.void to ctable objects
  (closes ticket #229 @eumiro)

- Do not try to flush when an carray is opened in 'r'ead-only mode.
  (closes #241 @FrancescAlted).

- Fix appending of object arrays to already existing carrays
  (closes #243 @cpcloud)

- Great modernization of setup.py by using new versioning and many
  other improvements (PR #239 @mindw).


Changes from 0.9.0 to 0.10.0
============================

- Fix pickle for in-memory carrays. (#193 #194 @dataisle @esc)

- Implement chunks iterator, which allows the following syntax
  ``for chunk_ in ca._chunks``, added "internal use" indicator to carray
  chunks attribute. (#153 @FrancescElies and @esc)

- Fix a memory leak and avoid copy in ``chunk.getudata``. (#201 #202 @esc)

- Fix the error message when trying to open a fresh ctable in an existing
  rootdir. (#191 @twiecki @esc)

- Solve #22 and be more specific about ``carray`` private methods.
  (#209 @FrancescElies @FrancescAlted)

- Implement context manager for ``carray`` and ``ctable``.
  (#135 #210 @FrancescElies and @esc)

- Fix handling and API for leftovers. (#72 #132 #211 #213 @FrancescElies @esc)

- Fix bug for incorrect leftover value. (#208 @waylonflinn)

- Documentation: document how to write extensions, update docstrings and
  mention the with statement / context manager. (#214 @FrancescElies)

- Various refactorings and cleanups. (#190 #198 #197 #199 #200)

- Fix bug creating carrays from transposed arrays without explicit dtype.
  (#217 #218 @sdvillal)


Changes from 0.8.1 to 0.9.0
===========================

- Implement ``purge``, which removes data for on-disk carrays. (#130 @esc)

- Implement ``addcol/delcol`` to properly handle on-disk ctable (#112/#151 @cpcloud @esc)

- Adding io-mode to the ``repr`` for carrays. (#124 @esc)

- Implement ``auto_flush`` which allows ctables to flush themselves during
  operations that modify (write) data.
  (#140 #152 @FrancescElies @CarstVaartjes @esc)

- Implement ``move`` for ctable, which allows disk-based carray to be moved
  (``mv``) into the root directory of the ctable.
  (#140 #152 #170 @FrancescElies @CarstVaartjes @esc)

- Distribute ``carray_ext.pxd`` as part of the package. (#159 @ARF)

- Add ``safe=`` keyword argument to control dtype/stride checking on append
  (#163 @mrocklin)

- Hold GIL during c-blosc compression/decompression, avoiding some segfaults
  (#166 @mrocklin)

- Fix ``dtype`` for multidimensional columns in a ctable (#136 #172 @alimanfoo)

- Fix to allow adding strings > len 1 to ctable (#178 @brentp)

- Sphinx based API documentation is now built from the docstrings in the Python
  sourcecode (#171 @esc)

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
