======================
Announcing bcolz 0.7.0
======================

What's new
==========

In this release, support for Python 3 has been added, new pandas and
HDF5/PyTables conversion, support for different compressors via latest
release of Blosc, and a new `iterblocks()` iterator.

``bcolz`` is a renaming of the ``carray`` project.  The new goals for
the project are to create a simple, yet flexible compressed containers,
that can live either on-disk or in-memory, and with some
high-performance iterators (like `iter()`, `where()`) for querying them.

For more detailed info, see the release notes in:
https://github.com/Blosc/bcolz/wiki/Release-Notes


What it is
==========

bcolz provides columnar and compressed data containers.  Column storage
allows for efficiently querying tables with a large number of columns.
It also allows for cheap addition and removal of column.  In addition,
bcolz objects are compressed by default for reducing memory/disk I/O
needs.  The compression process is carried out internally by Blosc, a
high-performance compressor that is optimized for binary data.

bcolz can use numexpr internally so as to accelerate many vector and
query operations (although it can use pure NumPy for doing so too).
numexpr optimizes the memory usage and use several cores for doing the
computations, so it is blazing fast.  Moreover, the carray/ctable
containers can be disk-based, and it is possible to use them for
seamlessly performing out-of-memory computations.

bcolz has minimal dependencies (NumPy), comes with an exhaustive test
suite and fully supports both 32-bit and 64-bit platforms.  Also, it is
typically tested on both UNIX and Windows operating systems.


Installing
==========

bcolz is in the PyPI repository, so installing it is easy:

$ pip install -U bcolz


Resources
=========

Visit the main bcolz site repository at:
http://github.com/Blosc/bcolz

Manual:
http://bcolz.blosc.org

Home of Blosc compressor:
http://blosc.org

User's mail list:
bcolz@googlegroups.com
http://groups.google.com/group/bcolz

License is the new BSD:
https://github.com/Blosc/bcolz/blob/master/LICENSES/BCOLZ.txt


----

  **Enjoy data!**


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
