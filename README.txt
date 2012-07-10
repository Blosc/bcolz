carray: A chunked data container that can be compressed in-memory
=================================================================

carray is a chunked container for numerical data.  Chunking allows for
efficient enlarging/shrinking of data container.  In addition, it can
also be compressed for reducing memory needs.  The compression process
is carried out internally by Blosc, a high-performance compressor that
is optimized for binary data.

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
already the case for some special cases now (2011), but will happen
more generally in a short future, when carray will be able to take
advantage of newer CPUs integrating more cores and wider vector units
(256 bit and more).

Building
--------

There are two differnt ways to build from sources, depending on how you obtained
them.

From tarball
~~~~~~~~~~~~

Assuming that you have NumPy, Cython and a C compiler installed, do:

$ python setup.py build_ext --inplace

From Git clone
~~~~~~~~~~~~~~

To build in a git clone you must use `paver <http://paver.github.com/paver/>`_
and then issue:

$ paver build


Testing
-------

After compiling, you can quickly check that the package is sane by
running:

$ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
$ export PYTHONPATH    (not needed on Windows)
$ python carray/tests/tests_all.py

Installing
----------

Installation, like building, depends on how you obtained the sources.

From tarball
~~~~~~~~~~~~

Install it as a typical Python package:

$ python setup.py install

From Git clone
~~~~~~~~~~~~~~

Again, using paver:

$ paver install

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
