carray: A chunked, compressed, data container (for memory and disk)
===================================================================


**Note:** This project is currently being developed as a persistent
layer for Blaze, with the BLZ codename.  You can find the sources
for BLZ over here:
https://github.com/ContinuumIO/blaze/tree/master/blaze/io/blz 

carray is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory/disk needs.  The compression
process is carried out internally by Blosc, a high-performance
compressor that is optimized for binary data.

carray can use numexpr internally so as to accelerate many vector and
query operations (although it can use pure NumPy for doing so too).
numexpr can use optimize the memory usage and use several cores for
doing the computations, so it is blazing fast.  Moreover, with the
introduction of a carray/ctable disk-based container (in version 0.5),
it can be used for seamlessly performing out-of-core computations.

Rational
--------

By using compression, you can deal with more data using the same
amount of memory.  In case you wonder: which is the price to pay in
terms of performance? you should know that nowadays memory access is
the most common bottleneck in many computational scenarios, and CPUs
spend most of its time waiting for data, and having data compressed in
memory can reduce the stress of the memory subsystem.

In other words, the ultimate goal for carray is not only reducing the
memory needs of large arrays, but also making carray operations to go
faster than using a traditional ndarray object from NumPy.  That is
already the case for some special cases now, but will happen more
generally in a short future, when carray will be able to take
advantage of newer CPUs integrating more cores and wider vector units.

Requisites
----------

- Python >= 2.6
- NumPy >= 1.5
- Cython >= 0.16

Building
--------

Assuming that you have the requisites and a C compiler installed, do:

$ python setup.py build_ext --inplace

Testing
-------

After compiling, you can quickly check that the package is sane by
running:

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
$ export PYTHONPATH    (not needed on Windows)
$ python carray/tests/test_all.py

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

Resources
---------

Visit the main carray site repository at:
http://github.com/FrancescAlted/carray

You can download a source package from:
http://carray.pytables.org/download

Manual:
http://carray.pytables.org/docs/manual

Home of Blosc compressor:
http://blosc.pytables.org

User's mail list:
carray@googlegroups.com
http://groups.google.com/group/carray

License
-------

Please see CARRAY.txt in LICENSES/ directory.

Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.


Francesc Alted
