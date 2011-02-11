carray: A chunked data container that can be compressed in-memory
=================================================================

carray is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory needs.  The compression process
is carried out internally by Blosc, a high-performance compressor that
is optimized for binary data.

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

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
$ export PYTHONPATH    (not needed on Windows)
$ python carray/tests/tests_all.py

Installing
----------

Install it as a typical Python package:

$ python setup.py install

Documentation
-------------

Please refer to the doc/ directory.  The HTML manual is in doc/html/,
and the PDF version is in doc/carray-manual.pdf.  Of course, you can
always access docstrings from the console (i.e. help(carray.ctable)).

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
