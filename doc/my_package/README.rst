Bcolz extension basic example
=============================
Install needed packages::

  $ pip install cython
  $ pip install numpy
  $ pip install bcolz

Build me
--------
Build the Bcolz Cython extension::

  $ python setup.py build_ext --inplace

Test me
-------

Start your python session::

  >>> import bcolz as bz
  >>> import my_extension.example_ext as my_mod
  >>> c = bz.carray([i for i in range(1000)], dtype='i8')
  >>> my_mod.my_function(c)
  499500

