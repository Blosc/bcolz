carray: a compressed in-memory data container
=============================================

carray is a container for data that can be compressed.  Nowadays
memory access is the most common bottleneck in many computational
scenarios, and CPUs spend most of its time waiting for data.

Having data compressed in-memory can reduce the stress of the memory
subsystem.  The net result is that carray operations can be faster
than using a traditional ndarray object from NumPy.

Building
--------

Assuiming that you have NumPy, Cython and a C compiler installed, do:

$ python setup.py build_ext --inplace

Testing
-------

After compiling, you can quickly check that the package is sane by
running:

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
$ export PYTHONPATH=.  (not needed on win)
$ python carray/tests/tests_all.py

Installing
----------

Install it as a typical Python package:

$ python setup.py install
