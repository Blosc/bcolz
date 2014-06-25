bcolz: columnar and compressed data containers
==============================================

**Note:** This is a renaming of the original **carray** project.

bcolz provides columnar and compressed data containers.  Column
storage allows for efficiently querying tables with a large number of
columns.  It also allows for cheap addition and removal of column.  In
addition, bcolz objects are compressed by default for reducing
memory/disk I/O needs.  The compression process is carried out
internally by Blosc, a high-performance compressor that is optimized
for binary data.

bcolz can use numexpr internally so as to accelerate many vector and
query operations (although it can use pure NumPy for doing so too).
numexpr can use optimize the memory usage and use several cores for
doing the computations, so it is blazing fast.  Moreover, with the
introduction of a carray/ctable disk-based container (in version 0.5),
it can be used for doing out-of-core computations transparently.

Rational
--------

By using compression, you can deal with more data using the same
amount of memory.  In case you wonder: which is the price to pay in
terms of performance? you should know that nowadays memory access is
the most common bottleneck in many computational scenarios, and CPUs
spend most of its time waiting for data, and having data compressed in
memory can reduce the stress of the memory subsystem.

In other words, the ultimate goal for bcolz is not only reducing the
memory needs of large arrays, but also making bcolz operations to go
faster than using a traditional ndarray object from NumPy.  That is
already the case for some special cases now, but will happen more
generally in a short future, when bcolz will be able to take advantage
of newer CPUs integrating more cores and wider vector units.

Requisites
----------

- Python >= 2.6
- NumPy >= 1.7
- Cython >= 0.20

Building
--------

Assuming that you have the requisites and a C compiler installed, do::

    $ python setup.py build_ext --inplace

Testing
-------

After compiling, you can quickly check that the package is sane by
running::

    $ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
    $ export PYTHONPATH    (not needed on Windows)
    $ python bcolz/tests/test_all.py

Installing
----------

Install it as a typical Python package::

    $ python setup.py install

Documentation
-------------

Please refer to the doc/ directory.  The HTML manual is in doc/html,
but of course, you can always access docstrings from the console
(i.e. help(bcolz.ctable)).

Also, you may want to look at the bench/ directory for some examples
of use.

Resources
---------

Visit the main bcolz site repository at:
http://github.com/Blosc/bcolz

Manual:
http://bcolz.blosc.org

Home of Blosc compressor:
http://blosc.org

User's mail list:
bcolz@googlegroups.com
http://groups.google.com/group/bcolz

License
-------

Please see BCOLZ.txt in LICENSES/ directory.

Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.

**Enjoy Data!**

Francesc Alted
