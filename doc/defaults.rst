.. _defaults:

-----------------------------
Defaults for bcolz operation
-----------------------------

You can tailor the behaviour of bcolz by changing the values of
certain some special top level variables whose defaults are listed
here.  You can change these values in two ways:

* In your program: the changes will be temporary.  For example::

    bcolz.defaults.eval_out_flavor = "numpy"

* Manually modify the ``defaults.py`` module of the bcolz package: the
  changes will be persistent.  For example, replace::

    defaults.eval_out_flavor = "carray"

  by::

    defaults.eval_out_flavor = "numpy"


List of default values
======================

.. py:attribute:: eval_out_flavor

    The flavor for the output object in :py:func:`eval`.  It can be 'carray'
    or 'numpy'.  Default is 'carray'.

.. py:attribute:: eval_vm

    The virtual machine to be used in computations (via :py:func:`eval`).
    It can be 'python' or 'numexpr'.  Default is 'numexpr',
    if installed.  If not, then the default is 'python'.


.. py:attribute:: cparams
   :noindex:

    The defaults for parameters used in compression (dict).  The
    default is {'clevel': 5, 'shuffle': True, 'cname': 'blosclz'}.

    See Also:
        :py:func:`cparams.setdefaults`
