.. _bcolz-defaults:

-----------------------------
Defaults for bcolz operation
-----------------------------

You can tailor the behaviour of bcolz by changing the values of
certain some special top level variables whose defaults are listed
here.  You can change these values in two ways:

* In your program: the changes will be temporary.

* In the ``defaults.py`` module of the bcolz package: the changes
  will be persistent.


List of default values
======================

.. py:attribute:: eval_out_flavor

    The flavor for the output object in :py :func:`eval`.  It can be 'carray'
    or 'numpy'.  Default is 'carray'.

.. py:attribute:: eval_vm

    The virtual machine to be used in computations (via :py :func:`eval`).
    It can be 'python' or 'numexpr'.  Default is 'numexpr',
    if installed.  If not, then the default is 'python'.


