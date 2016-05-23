-----------------
Library Reference
-----------------

.. currentmodule:: bcolz

First level variables
=====================

.. py:attribute:: __version__

    The version of the bcolz package.

.. py:attribute:: dask_here

    Whether the minimum version of dask has been detected.

.. py:attribute:: min_dask_version

    The minimum version of dask needed (dask is optional).

.. py:attribute:: min_numexpr_version

    The minimum version of numexpr needed (numexpr is optional).

.. py:attribute:: ncores

    The number of cores detected.

.. py:attribute:: numexpr_here

    Whether the minimum version of numexpr has been detected.


Top level classes
=================

.. autoclass:: cparams
   :members: setdefaults

.. autoclass:: bcolz.attrs.attrs

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

.. this is needed because the ctable class resides in the ctable module
.. currentmodule:: bcolz.ctable

.. autoclass:: ctable
   :members:
