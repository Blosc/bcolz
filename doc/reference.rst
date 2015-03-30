-----------------
Library Reference
-----------------

.. currentmodule:: bcolz

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
=================

.. autoclass:: cparams
   :members: setdefaults


Also, see the :py:class:`carray` and :py:class:`ctable` classes below.

.. _top-level-constructors:

Top level functions
===================

.. autofunction:: arange

.. autofunction:: eval

.. autofunction:: fill

.. autofunction:: fromiter

.. autofunction:: iterblocks

.. autofunction:: ones

.. autofunction:: zeros

.. autofunction:: open

.. autofunction:: walk


Top level printing functions
============================

.. py:function:: array2string(a, max_line_width=None, precision=None, suppress_small=None, separator=' ', prefix="", style=repr, formatter=None)

    Return a string representation of a carray/ctable object.

    This is the same function than in NumPy.  Please refer to NumPy
    documentation for more info.

    See Also:
      :py:func:`set_printoptions`, :py:func:`get_printoptions`

.. py:function:: get_printoptions()

    Return the current print options.

    This is the same function than in NumPy.  For more info, please
    refer to the NumPy documentation.

    See Also:
      :py:func:`array2string`, :py:func:`set_printoptions`

.. py:function:: set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, formatter=None)

    Set printing options.

    These options determine the way floating point numbers in carray
    objects are displayed.  This is the same function than in NumPy.
    For more info, please refer to the NumPy documentation.

    See Also:
      :py:func:`array2string`, :py:func:`get_printoptions`

Utility functions
=================

.. autofunction:: set_nthreads

.. autofunction:: blosc_set_nthreads

.. autofunction:: detect_number_of_cores

.. autofunction:: blosc_version

.. autofunction:: print_versions

.. autofunction:: test

The carray class
================

.. autoclass:: carray
   :members:
   :special-members: __getitem__, __setitem__

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
