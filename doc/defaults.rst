-----------------------------
Defaults for carray operation
-----------------------------

You can tailor the behaviour of carray by changing the values of
certain defaults listed here.  You can change the values in two ways:

* In your program: the changes will be temporary.

* In the ``defaults.py`` module of the carray package: the changes
  will be persistent.


List of default values
======================

.. py:attribute:: eval_vm

    The virtual machine to be used in computations (via `eval`).  It
    can be "numexpr" or "python".


.. py:attribute:: eval_out_flavor

    The flavor for the output object in `eval()`.  It can be 'carray'
    or 'numpy'.


