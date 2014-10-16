-----------------
Library Reference
-----------------

First level variables
=====================

.. py:attribute:: __version__

    The version of the bcolz package.

.. py:attribute:: min_numexpr_version

    The minimum version of numexpr needed (numexpr is optional).

.. py:attribute:: ncores

    The number of cores detected.

.. py:attribute:: numexpr_here

    Whether minimum version of numexpr has been detected.


Top level classes
===================

.. py:class:: cparams(clevel=None, shuffle=None, cname=None)

    Class to host parameters for compression and other filters.

    Parameters:
      clevel : int (0 <= clevel < 10)
        The compression level.
      shuffle : bool
        Whether the shuffle filter is active or not.
      cname : string ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib')
        Select the compressor to use inside Blosc.

    In case some of the parameters are not passed, they will be set to
    a default (see `setdefaults()` method).

    See Also:
        :py:func:`cparams.setdefaults`


  .. py:method:: setdefaults(clevel=None, shuffle=None, cname=None)

    Change the defaults for `clevel`, `shuffle` and `cname` params.

    Parameters:
      clevel : int (0 <= clevel < 10)
        The compression level.
      shuffle : bool
        Whether the shuffle filter is active or not.
      cname : string ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib')
        Select the compressor to use inside Blosc.

    If this method is not called, the defaults will be set as in
    defaults.py (``{clevel=5, shuffle=True, cname='blosclz'}``).


Also, see the :py:class:`carray` and :py:class:`ctable` classes below.

.. _top-level-constructors:

Top level functions
=====================

.. py:function:: array2string(a, max_line_width=None, precision=None, suppress_small=None, separator=' ', prefix="", style=repr, formatter=None)

    Return a string representation of a carray/ctable object.

    This is the same function than in NumPy.  Please refer to NumPy
    documentation for more info.

    See Also:
      :py:func:`set_printoptions`, :py:func:`get_printoptions`


.. py:function:: arange([start,] stop[, step,], dtype=None, **kwargs)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start,
    stop)`` (in other words, the interval including `start` but
    excluding `stop`).  For integer arguments the function is
    equivalent to the Python built-in `range
    <http://docs.python.org/lib/built-in-funcs.html>`_ function, but
    returns a carray rather than a list.

    Parameters:
      start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
      stop : number
        End of interval.  The interval does not include this value.
      step : number, optional
        Spacing between values.  For any output `out`, this is the
        distance between two adjacent values, ``out[i+1] - out[i]``.
        The default step size is 1.  If `step` is specified, `start`
        must also be given.
      dtype : dtype
        The type of the output array.  If `dtype` is not given, infer
        the data type from the other input arguments.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

.. py:function:: eval(expression, vm=None, out_flavor=None, user_dict=None, **kwargs)

    Evaluate an `expression` and return the result.

    Parameters:
      expression : string
        A string forming an expression, like '2*a+3*b'. The values for
        'a' and 'b' are variable names to be taken from the calling
        function's frame.  These variables may be scalars, carrays or
        NumPy arrays.
      vm : string
        The virtual machine to be used in computations.  It can be 'numexpr'
        or 'python'.  The default is to use 'numexpr' if it is installed.
      out_flavor : string
        The flavor for the `out` object.  It can be 'carray' or 'numpy'.
      user_dict : dict
        An user-provided dictionary where the variables in expression
        can be found by name.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray object
        The outcome of the expression.  You can tailor the
        properties of this carray by passing additional arguments
        supported by carray constructor in `kwargs`.

.. py:function:: fill(shape, dflt=None, dtype=float, **kwargs)

    Return a new carray object of given shape and type, filled with `dflt`.

    Parameters:
      shape : int
        Shape of the new array, e.g., ``(2,3)``.
      dflt : Python or NumPy scalar
        The value to be used during the filling process.  If None, values are
        filled with zeros.  Also, the resulting carray will have this value as
        its `dflt` value.
      dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray
        Array filled with `dflt` values with the given shape and dtype.

    See Also:
      :py:func:`zeros`, :py:func:`ones`

.. py:function:: fromiter(iterable, dtype, count, **kwargs)

    Create a carray/ctable from an `iterable` object.

    Parameters:
      iterable : iterable object
        An iterable object providing data for the carray.
      dtype : numpy.dtype instance
        Specifies the type of the outcome object.
      count : int
        The number of items to read from iterable. If set to -1, means
        that the iterable will be used until exhaustion (not
        recommended, see note below).
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray/ctable constructors.

    Returns:
      out : a carray/ctable object

    Notes:
      Please specify `count` to both improve performance and to save
      memory.  It allows `fromiter` to avoid looping the iterable
      twice (which is slooow).  It avoids memory leaks to happen too
      (which can be important for large iterables).

.. py:function:: get_printoptions()

    Return the current print options.

    This is the same function than in NumPy.  For more info, please
    refer to the NumPy documentation.

    See Also:
      :py:func:`array2string`, :py:func:`set_printoptions`


.. py:function:: iterblocks(blen=None, start=0, stop=None)

    Iterate over a `cobj` (carray/ctable) in blocks of size `blen`.

    Parameters:
      cobj : carray/ctable object
        The bcolz object to be iterated over.
      blen : int
        The length of the block that is returned.  The default is the
        chunklen, or for a ctable, the minimum of the different column
        chunklens.
      start : int
        Where the iterator starts.  The default is to start at the beginning.
      stop : int
        Where the iterator stops. The default is to stop at the end.

    Returns:
      out : iterable
        This iterable returns buffers as NumPy arays of homogeneous or
        structured types, depending on whether `cobj` is a carray or a
        ctable object.

    See Also:
      :py:meth:`ctable.whereblocks`


.. py:function:: ones(shape, dtype=float, **kwargs)

    Return a new carray object of given shape and type, filled with ones.

    Parameters:
      shape : int
        Shape of the new array, e.g., ``(2,3)``.
      dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray
        Array of ones with the given shape and dtype.

    See Also:
      :py:func:`fill`, :py:func:`ones`


.. py:function:: open(rootdir, mode='a')

    Open a disk-based carray/ctable.

    Parameters:
      rootdir : pathname (string)
        The directory hosting the carray/ctable object.
      mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns:
      out : a carray/ctable object or None (if not objects are found)


.. py:function:: print_versions()

    Print all the versions of packages that bcolz relies on.


.. py:function:: set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)

    Set printing options.

    These options determine the way floating point numbers in carray
    objects are displayed.  This is the same function than in NumPy.
    For more info, please refer to the NumPy documentation.

    See Also:
      :py:func:`array2string`, :py:func:`get_printoptions`


.. py:function:: zeros(shape, dtype=float, **kwargs)

    Return a new carray object of given shape and type, filled with zeros.

    Parameters:
      shape : int
        Shape of the new array, e.g., ``(2,3)``.
      dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray
        Array of zeros with the given shape and dtype.

    See Also:
      :py:func:`fill`, :py:func:`zeros`


.. py:function:: walk(dir, classname=None, mode='a')

    Recursively iterate over carray/ctable objects hanging from `dir`.

    Parameters:
      dir : string
        The directory from which the listing starts.
      classname : string
        If specified, only object of this class are returned.  The values
        supported are 'carray' and 'ctable'.
      mode : string
        The mode in which the object should be opened.

    Returns:
      out : iterator
        Iterator over the objects found.



Utility functions
=================

.. py:function:: blosc_set_nthreads(nthreads)

    Sets the number of threads that Blosc can use.

    Parameters:
      nthreads : int
        The desired number of threads to use.

    Returns:
      out : int
        The previous setting for the number of threads.

.. py:function:: blosc_version()

    Return the version of the Blosc library.

.. py:function:: detect_number_of_cores()

    Return the number of cores on a system.

.. py:function:: set_nthreads(nthreads)

    Sets the number of threads to be used during bcolz operation.

    This affects to both Blosc and Numexpr (if available).

    Parameters:
      nthreads : int
        The number of threads to be used during bcolz operation.

    Returns:
      out : int
        The previous setting for the number of threads.

    See Also:
      :py:func:`blosc_set_nthreads`


.. py:function:: test(verbose=False, heavy=False)

    Run all the tests in the test suite.

    If `verbose` is set, the test suite will emit messages with full
    verbosity (not recommended unless you are looking into a certain
    problem).

    If `heavy` is set, the test suite will be run in *heavy* mode (you
    should be careful with this because it can take a lot of time and
    resources from your computer).


The carray class
================

.. py:class:: carray(array, cparams=None, dtype=None, dflt=None, expectedlen=None, chunklen=None, rootdir=None, mode='a')

  A compressed and enlargeable in-memory data container.

  `carray` exposes a series of methods for dealing with the compressed
  container in a NumPy-like way.

  Parameters:
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
      A guess on the expected length of this carray.  This will serve to
      decide the best `chunklen` used for compression and memory I/O
      purposes.
    chunklen : int, optional
      The number of items that fits on a chunk.  By specifying it you can
      explicitly set the chunk size used for compression and memory I/O.
      Only use it if you know what are you doing.
  rootdir : str, optional
      The directory where all the data and metadata will be stored.  If
      specified, then the carray object will be disk-based (i.e. all chunks
      will live on-disk, not in memory) and persistent (i.e. it can be
      restored in other session, e.g. via the `open()` top level function).
  mode : str, optional
      The mode that a *persistent* carray should be created/opened.  The
      values can be:

        * 'r' for read-only
        * 'w' for read/write.  During carray creation, the `rootdir`
          will be removed if it exists.  During carray opening, the
          carray will be resized to 0.
        * 'a' for append (possible data inside `rootdir` will not be removed).

.. _carray-attributes:

carray attributes
-----------------

  .. py:attribute:: attrs

    Accessor for attributes in carray objects.

    This class behaves very similarly to a dictionary, and attributes
    can be appended in the typical way::

       attrs['myattr'] = value

    And can be retrieved similarly::

       value = attrs['myattr']

    Attributes can be removed with::

       del attrs['myattr']

    This class also honors the `__iter__` and `__len__` special
    functions.  Moreover, a `getall()` method returns all the
    attributes as a dictionary.

    CAVEAT: The values should be able to be serialized with JSON for
    persistence.

  .. py:attribute:: cbytes

    The compressed size of this object (in bytes).

  .. py:attribute:: chunklen

    The number of items that fits into a chunk.

  .. py:attribute:: cparams

    The compression parameters for this object.

  .. py:attribute:: dflt

    The value to be used when enlarging the carray.

  .. py:attribute:: dtype

    The NumPy dtype for this object.

  .. py:attribute:: len

    The length of this object.

  .. py:attribute:: nbytes

    The original (uncompressed) size of this object (in bytes).

  .. py:attribute:: ndim

    The number of dimensions of this object (in bytes).

  .. py:attribute:: shape

    The shape of this object.

  .. py:attribute:: size

    The size of this object.


carray methods
--------------

  .. py:method:: append(array)

    Append a numpy `array` to this instance.

    Parameters:
      array : NumPy-like object
        The array to be appended.  Must be compatible with shape and type of
        the carray.


  .. py:method:: copy(**kwargs)

    Return a copy of this object.

    Parameters:
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns:
      out : carray object
        The copy of this object.

    See Also:
      :py:meth:`view`


  .. py:method:: flush()

    Flush data in internal buffers to disk.

    This call should typically be done after performing modifications
    (__settitem__(), append()) in persistence mode.  If you don't do this, you
    risk losing part of your modifications.


  .. py:method:: free_cachemem()

    Get rid of internal caches to free memory.

    This call can typically be made after reading from a
    carray/ctable so as to free the memory used internally to
    cache data blocks/chunks.


  .. py:method:: iter(start=0, stop=None, step=1, limit=None, skip=0)

    Iterator with `start`, `stop` and `step` bounds.

    Parameters:
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
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterator

    See Also:
      :py:meth:`where`, :py:meth:`wheretrue`


  .. py:method:: reshape(newshape)

    Returns a new carray containing the same data with a new shape.

    Parameters:
      newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.

    Returns:
      reshaped_array : carray
        A copy of the original carray.


  .. py:method:: resize(nitems)

    Resize the instance to have `nitems`.

    Parameters:
      nitems : int
        The final length of the object.  If `nitems` is larger than
        the actual length, new items will appended using `self.dflt`
        as filling values.


  .. py:method:: sum(dtype=None)

    Return the sum of the array elements.

    Parameters:
      dtype : NumPy dtype
        The desired type of the output.  If ``None``, the dtype of
        `self` is used.  An exception is when `self` has an integer
        type with less precision than the default platform integer.
        In that case, the default platform integer is used instead
        (NumPy convention).

    Return value:
      out : NumPy scalar with `dtype`

  .. py:method:: trim(nitems)

    Remove the trailing `nitems` from this instance.

    Parameters:
      nitems : int
        The number of trailing items to be trimmed.

    See Also:
      :py:meth:`append`


  .. py:method:: view()

    Create a light weight view of the data in the original carray.

    Returns:
      out : carray object
        The view of this object.

    See Also:
      :py:meth:`copy`


  .. py:method:: where(boolarr, limit=None, skip=0)

    Iterator that returns values of this object where `boolarr` is true.

    This is currently only useful for boolean carrays that are unidimensional.

    Parameters:
      boolarr : a carray or NumPy array of boolean type
        The boolean values.
      limit : int
        A maximum number of elements to return.  The default is return
        everything.
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterator

    See Also:
      :py:meth:`iter`, :py:meth:`wheretrue`


  .. py:method:: wheretrue(limit=None, skip=0)

    Iterator that returns indices where this object is true.

    This is currently only useful for boolean carrays that are unidimensional.

    Parameters:
      limit : int
        A maximum number of elements to return.  The default is return
        everything.
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterator

    See Also:
      :py:meth:`iter`, :py:meth:`where`


carray special methods
----------------------

  .. py:method::  __getitem__(key):

    x.__getitem__(key) <==> x[key]

    Returns values based on `key`.  All the functionality of
    ``ndarray.__getitem__()`` is supported (including fancy indexing),
    plus a special support for expressions:

    Parameters:
      key : string
        It will be interpret as a boolean expression (computed via
        `eval`) and the elements where these values are true will be
        returned as a NumPy array.

    See Also:
      eval


  .. py:method::  __setitem__(key, value):

    x.__setitem__(key, value) <==> x[key] = value

    Sets values based on `key`.  All the functionality of
    ``ndarray.__setitem__()`` is supported (including fancy indexing),
    plus a special support for expressions:

    Parameters:
      key : string
        It will be interpret as a boolean expression (computed via
        `eval`) and the elements where these values are true will be
        set to `value`.

    See Also:
      eval


The ctable class
================

.. py:class:: ctable(columns, names=None, **kwargs)

    This class represents a compressed, column-wise, in-memory table.

    Create a new ctable from `columns` with optional `names`.

    Parameters:
      columns : tuple or list of column objects
        The list of column data to build the ctable object.  This can also be
        a pure NumPy structured array.  A list of lists or tuples is valid
        too, as long as they can be converted into carray objects.
      names : list of strings or string
        The list of names for the columns.  Alternatively, it can be
        specified as a string such as 'f0 f1' or 'f0, f1'.  If not
        passed, the names will be chosen as 'f0' for the top column,
        'f1' for the second and so on so forth (NumPy convention).
      kwargs : list of parameters or dictionary
        Allows to pass additional arguments supported by carray
        constructors in case new carrays need to be built.

    Notes:
      Columns passed as carrays are not be copied, so their settings
      will stay the same, even if you pass additional arguments
      (cparams, chunklen...).


ctable attributes
-----------------

  .. py:attribute:: attrs

    Accessor for attributes in ctable objects.

    See :py:attr:`bcolz.attrs` for a full description.

  .. py:attribute:: cbytes

    The compressed size of this object (in bytes).

  .. py:attribute:: cols

    The ctable columns accessor.

  .. py:attribute:: cparams

    The compression parameters for this object.

  .. py:attribute:: dtype

    The NumPy dtype for this object.

  .. py:attribute:: len

    The length of this object.

  .. py:attribute:: names

   The names of the columns (list).

  .. py:attribute:: nbytes

    The original (uncompressed) size of this object (in bytes).

  .. py:attribute:: ndim

    The number of dimensions of this object (in bytes).

  .. py:attribute:: shape

    The shape of this object.

  .. py:attribute:: size

    The size of this object.


ctable methods
--------------

  .. py:method:: addcol(newcol, name=None, pos=None, **kwargs)

    Add a new `newcol` object as column.

    Parameters:
      newcol : carray, ndarray, list or tuple
        If a carray is passed, no conversion will be carried out.
        If conversion to a carray has to be done, `kwargs` will
        apply.
      name : string, optional
        The name for the new column.  If not passed, it will
        receive an automatic name.
      pos : int, optional
        The column position.  If not passed, it will be appended
        at the end.
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Notes:
      You should not specify both `name` and `pos` arguments,
      unless they are compatible.

    See Also:
      :py:func:`delcol`


  .. py:method:: append(cols)

    Append `cols` to this ctable.

    Parameters:
      cols : list/tuple of scalar values, NumPy arrays or carrays
        It also can be a NumPy record, a NumPy recarray, or
        another ctable.


  .. py:method:: copy(**kwargs)

    Return a copy of this ctable.

    Parameters:
      kwargs : list of parameters or dictionary
        Any parameter supported by the carray/ctable constructor.

    Returns:
      out : ctable object
        The copy of this ctable.

  .. py:method:: delcol(name=None, pos=None)

    Remove the column named `name` or in position `pos`.

    Parameters:
      name: string, optional
        The name of the column to remove.
      pos: int, optional
        The position of the column to remove.

    Notes:
      You must specify at least a `name` or a `pos`.  You should
      not specify both `name` and `pos` arguments, unless they
      are compatible.

    See Also:
      :py:func:`addcol`


  .. py:method:: eval(expression, **kwargs)

    Evaluate the `expression` on columns and return the result.

    Parameters:
      expression : string
        A string forming an expression, like '2*a+3*b'. The values
        for 'a' and 'b' are variable names to be taken from the
        calling function's frame.  These variables may be column
        names in this table, scalars, carrays or NumPy arrays.
      kwargs : list of parameters or dictionary
        Any parameter supported by the `eval()` top level function.

    Returns:
      out : carray object
        The outcome of the expression.  You can tailor the
        properties of this carray by passing additional arguments
        supported by carray constructor in `kwargs`.

    See Also:
      :py:func:`eval` (top level function)


  .. py:method:: flush()

    Flush data in internal buffers to disk.

    This call should typically be done after performing modifications
    (__settitem__(), append()) in persistence mode.  If you don't do this, you
    risk losing part of your modifications.


  .. py:method:: free_cachemem()

    Get rid of internal caches to free memory.

    This call can typically be made after reading from a
    carray/ctable so as to free the memory used internally to
    cache data blocks/chunks.


  .. py:staticmethod:: fromdataframe(df, **kwargs)

    Return a ctable object out of a pandas dataframe.

    Parameters:
      df : DataFrame
        A pandas dataframe
      kwargs : list of parameters or dictionary
        Any parameter supported by the ctable constructor.

    Returns:
      out : ctable object
        A ctable filled with values from `df`.

    Note:
      The 'object' dtype will be converted into a 'S'tring type, if possible.
      This allows for much better storage savings in bcolz.

    See Also:
      :py:meth:`todataframe`


  .. py:staticmethod:: fromhdf5(filepath, nodepath='/ctable', **kwargs)

    Return a ctable object out of a compound HDF5 dataset (PyTables Table).

    Parameters:
      filepath : string
        The path of the HDF5 file.
      nodepath : string
        The path of the node inside the HDF5 file.
      kwargs : list of parameters or dictionary
        Any parameter supported by the ctable constructor.

    Returns:
      out : ctable object
        A ctable filled with values from the HDF5 node.

    See Also:
      :py:meth:`tohdf5`


  .. py:method:: iter(start=0, stop=None, step=1, outcols=None, limit=None, skip=0)

    Iterator with `start`, `stop` and `step` bounds.

    Parameters:
      start : int
        The starting item.
      stop : int
        The item after which the iterator stops.
      step : int
        The number of items incremented during each iteration.  Cannot be
        negative.
      outcols : list of strings or string
        The list of column names that you want to get back in results.
        Alternatively, it can be specified as a string such as 'f0 f1'
        or 'f0, f1'.  If None, all the columns are returned.  If the
        special name 'nrow__' is present, the number of row will be
        included in output.
      limit : int
        A maximum number of elements to return.  The default is return
        everything.
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterable

    See Also:
      :py:func:`ctable.where`

  .. py:method:: resize(nitems)

    Resize the instance to have `nitems`.

    Parameters:
      nitems : int
        The final length of the instance.  If `nitems` is larger than the
        actual length, new items will appended using `self.dflt` as
        filling values.


  .. py:method:: todataframe(columns=None, orient='columns')

    Return a pandas dataframe out of this object.

    Parameters:
      columns : sequence of column labels, optional
        Must be passed if orient='index'.
      orient : {'columns', 'index'}, default 'columns'
        The "orientation" of the data. If the keys of the input correspond
        to column labels, pass 'columns' (default). Otherwise if the keys
        correspond to the index, pass 'index'.

    Returns:
      out : DataFrame
        A pandas DataFrame filled with values from this object.

    See Also:
      :py:meth:`fromdataframe`


  .. py:method:: tohdf5(filepath, nodepath='/ctable', mode='w', cparams=None, cname=None)

    Write this object into an HDF5 file.

    Parameters:
      filepath : string
        The path of the HDF5 file.
      nodepath : string
        The path of the node inside the HDF5 file.
      mode : string
        The mode to open the PyTables file.  Default is 'w'rite mode.
      cparams : cparams object
        The compression parameters.  The defaults are the same than for
        the current bcolz environment.
      cname : string
        Any of the compressors supported by PyTables (e.g. 'zlib').  The
        default is to use 'blosc' as meta-compressor in combination with
        one of its compressors (see `cparams` parameter above).

    See Also:
       :py:meth:`fromhdf5`


  .. py:method:: trim(nitems)

    Remove the trailing `nitems` from this instance.

    Parameters:
      nitems : int
        The number of trailing items to be trimmed.

    See Also:
      :py:meth:`ctable.append`


  .. py:method:: where(expression, outcols=None, limit=None, skip=0)

    Iterate over rows where `expression` is true.

    Parameters:
      expression : string or carray
        A boolean Numexpr expression or a boolean carray.
      outcols : list of strings or string
        The list of column names that you want to get back in results.
        Alternatively, it can be specified as a string such as 'f0 f1'
        or 'f0, f1'.  If None, all the columns are returned.  If the
        special name 'nrow__' is present, the number of row will be
        included in output.
      limit : int
        A maximum number of elements to return.  The default is return
        everything.
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterable
        This iterable returns rows as NumPy structured types (i.e. they
        support being mapped either by position or by name).

    See Also:
      :py:meth:`ctable.iter`


  .. py:method:: whereblocks(expression, blen=None, outfields=None, limit=None, skip=0)

    Iterate over the rows that fullfill the `expression` condition on
    this ctable, in blocks of size `blen`.

    Parameters:
      expression : string or carray
        A boolean Numexpr expression or a boolean carray.
      blen : int
        The length of the block that is returned.  The default is the
        chunklen, or for a ctable, the minimum of the different column
        chunklens.
      outfields : list of strings or string
        The list of column names that you want to get back in results.
        Alternatively, it can be specified as a string such as 'f0 f1' or
        'f0, f1'.
      limit : int
        A maximum number of elements to return.  The default is return
        everything.
      skip : int
        An initial number of elements to skip.  The default is 0.

    Returns:
      out : iterable
        This iterable returns buffers as NumPy arrays made of
        structured types (or homogeneous ones in case `outfields` is a
        single field.

    See Also:
      :py:func:`iterblocks`



ctable special methods
----------------------

  .. py:method::  __getitem__(key):

    x.__getitem__(y) <==> x[y]

    Returns values based on `key`.  All the functionality of
    ``ndarray.__getitem__()`` is supported (including fancy indexing),
    plus a special support for expressions:

    Parameters:
      key : string
        The corresponding ctable column name will be returned.  If not
        a column name, it will be interpret as a boolean expression
        (computed via `ctable.eval`) and the rows where these values are
        true will be returned as a NumPy structured array.

    See Also:
      :py:meth:`ctable.eval`

  .. py:method::  __setitem__(key, value):

    x.__setitem__(key, value) <==> x[key] = value

    Sets values based on `key`.  All the functionality of
    ``ndarray.__setitem__()`` is supported (including fancy indexing),
    plus a special support for expressions:

    Parameters:
      key : string
        The corresponding ctable column name will be set to `value`.
        If not a column name, it will be interpret as a boolean
        expression (computed via `ctable.eval`) and the rows where these
        values are true will be set to `value`.

    See Also:
      :py:meth:`ctable.eval`
