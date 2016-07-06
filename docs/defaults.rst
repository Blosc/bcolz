.. _defaults:

-----------------------------
Defaults for bcolz operation
-----------------------------

You can tailor the behaviour of bcolz by changing the values of
certain some special top level variables whose defaults are listed
here.  You can change these values in two ways:

* In your program: the changes will be temporary.  For example::

    bcolz.defaults.out_flavor = "numpy"

* Manually modify the ``defaults.py`` module of the bcolz package: the
  changes will be persistent.  For example, replace::

    defaults.out_flavor = "bcolz"

  by::

    defaults.out_flavor = "numpy"

Generally, only the former is needed.

Defaults in contexts
====================

bcolz allows to set short-lived defaults in contexts.  For example::

   with bcolz.defaults_ctx(vm="python", cparams=bcolz.cparams(clevel=0)):
      cout = bcolz.eval("(x + 1) < 0")

means that the `bcolz.eval` operation will be made using a "python"
virtual machine and no compression for the `cout` output.

List of default values
======================

.. py:attribute:: out_flavor

    The flavor for the output object in :py:func:`eval` and others
    that call this indirectly.  It can be 'bcolz' or 'numpy'.  Default
    is 'bcolz'.

.. py:attribute:: vm

    The virtual machine to be used in computations (via
    :py:func:`eval`).  It can be 'python', 'numexpr' or 'dask'.
    Default is 'numexpr', if installed.  If not, 'dask' is used, if
    installed.  And if neither of these are installed, then the
    'python' interpreter is used (via numpy).


.. py:attribute:: cparams
   :noindex:

    The defaults for parameters used in compression (dict).  The
    default is {'clevel': 5, 'shuffle': True, 'cname': 'lz4',
    quantize: 0}.

    See Also:
        :py:func:`cparams.setdefaults`
