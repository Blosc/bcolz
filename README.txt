carray: a compressed and enlargeable in-memory data container
=============================================================

carray is a container for numerical data that can be compressed
in-memory.  The compresion process is carried out internally by Blosc,
a high-performance compressor that is optimized for binary data.

Rational
--------

Nowadays memory access is the most common bottleneck in many
computational scenarios, and CPUs spend most of its time waiting for
data.

Having data compressed in memory can reduce the stress of the memory
subsystem.  The net result is that carray operations can be faster
than using a traditional ndarray object from NumPy.

Building
--------

Assuming that you have NumPy, Cython and a C compiler installed, do:

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

Documentation
-------------

Please refer to USAGE.txt.  In case you want an accurate description
of the complete API, you will have to browse it from docstrings.
Start from `carray` package and continue reading the proposed links.

Also, you may want to look at the bench/ directory for some examples
of use.

User's mail list
----------------

Feel free to send questions related with carray to:

carray@googlegroups.com
http://groups.google.com/group/carray

License
-------

Please see CARRAY.txt in LICENSES/ directory.


Francesc Alted
