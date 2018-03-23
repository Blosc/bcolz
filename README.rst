bcolz: columnar and compressed data containers
==============================================

.. image:: https://badges.gitter.im/Blosc/bcolz.svg
   :alt: Join the chat at https://gitter.im/Blosc/bcolz
   :target: https://gitter.im/Blosc/bcolz?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

:Version: |version|
:Travis CI: |travis|
:Appveyor: |appveyor|
:Coveralls: |coveralls|
:And...: |powered|

.. |version| image:: https://img.shields.io/pypi/v/bcolz.png
        :target: https://pypi.python.org/pypi/bcolz

.. |travis| image:: https://img.shields.io/travis/Blosc/bcolz.png
        :target: https://travis-ci.org/Blosc/bcolz

.. |appveyor| image:: https://img.shields.io/appveyor/ci/FrancescAlted/bcolz.png
        :target: https://ci.appveyor.com/project/FrancescAlted/bcolz/branch/master

.. |powered| image:: http://b.repl.ca/v1/Powered--By-Blosc-blue.png
        :target: http://blosc.org

.. |coveralls| image:: https://coveralls.io/repos/Blosc/bcolz/badge.png
        :target: https://coveralls.io/r/Blosc/bcolz


.. image:: docs/bcolz.png

bcolz provides columnar, chunked data containers that can be
compressed either in-memory and on-disk.  Column storage allows for
efficiently querying tables, as well as for cheap column addition and
removal.  It is based on `NumPy <http://www.numpy.org>`_, and uses it
as the standard data container to communicate with bcolz objects, but
it also comes with support for import/export facilities to/from
`HDF5/PyTables tables <http://www.pytables.org>`_ and `pandas
dataframes <http://pandas.pydata.org>`_.

bcolz objects are compressed by default not only for reducing
memory/disk storage, but also to improve I/O speed.  The compression
process is carried out internally by `Blosc <http://blosc.org>`_, a
high-performance, multithreaded meta-compressor that is optimized for
binary data (although it works with text data just fine too).

bcolz can also use `numexpr <https://github.com/pydata/numexpr>`_
internally (it does that by default if it detects numexpr installed)
or `dask <https://github.com/dask/dask>`_ so as to accelerate many
vector and query operations (although it can use pure NumPy for doing
so too).  numexpr/dask can optimize the memory usage and use
multithreading for doing the computations, so it is blazing fast.
This, in combination with carray/ctable disk-based, compressed
containers, can be used for performing out-of-core computations
efficiently, but most importantly *transparently*.

Just to whet your appetite, here it is an example with real data, where
bcolz is already fulfilling the promise of accelerating memory I/O by
using compression:

http://nbviewer.ipython.org/github/Blosc/movielens-bench/blob/master/querying-ep14.ipynb


Rationale
---------

By using compression, you can deal with more data using the same
amount of memory, which is very good on itself.  But in case you are
wondering about the price to pay in terms of performance, you should
know that nowadays memory access is the most common bottleneck in many
computational scenarios, and that CPUs spend most of its time waiting
for data.  Hence, having data compressed in memory can reduce the
stress of the memory subsystem as well.

Furthermore, columnar means that the tabular datasets are stored
column-wise order, and this turns out to offer better opportunities to
improve compression ratio.  This is because data tends to expose more
similarity in elements that sit in the same column rather than those
in the same row, so compressors generally do a much better job when
data is aligned in such column-wise order.  In addition, when you have
to deal with tables with a large number of columns and your operations
only involve some of them, a columnar-wise storage tends to be much
more effective because minimizes the amount of data that travels to
CPU caches.

So, the ultimate goal for bcolz is not only reducing the memory needs
of large arrays/tables, but also making bcolz operations to go faster
than using a traditional data container like those in NumPy or Pandas.
That is actually already the case in some real-life scenarios (see the
notebook above) but that will become pretty more noticeable in
combination with forthcoming, faster CPUs integrating more cores and
wider vector units.

Requisites
----------

- Python >= 2.7 and >= 3.5
- NumPy >= 1.8
- Cython >= 0.22 (just for compiling the beast)
- C-Blosc >= 1.8.0 (optional, as the internal Blosc will be used by default)

Optional:

- numexpr >= 2.5.2
- dask >= 0.9.0
- pandas
- tables (pytables)

Building
--------

There are different ways to compile bcolz, depending if you want to
link with an already installed Blosc library or not.

Compiling with an installed Blosc library (recommended)
.......................................................

Python and Blosc-powered extensions have a difficult relationship when
compiled using GCC, so this is why using an external C-Blosc library is
recommended for maximum performance (for details, see
https://github.com/Blosc/python-blosc/issues/110).

Go to https://github.com/Blosc/c-blosc/releases and download and
install the C-Blosc library.  Then, you can tell bcolz where is the
C-Blosc library in a couple of ways:

Using an environment variable:

.. code-block:: console

    $ BLOSC_DIR=/usr/local     (or "set BLOSC_DIR=\blosc" on Win)
    $ export BLOSC_DIR         (not needed on Win)
    $ python setup.py build_ext --inplace

Using a flag:

.. code-block:: console

    $ python setup.py build_ext --inplace --blosc=/usr/local

Compiling without an installed Blosc library
............................................

bcolz also comes with the Blosc sources with it so, assuming that you
have a C++ compiler installed, do:

.. code-block:: console

    $ python setup.py build_ext --inplace

That's all.  You can proceed with testing section now.

Note: The requirement for the C++ compiler is just for the Snappy
dependency.  The rest of the other components of Blosc are pure C
(including the LZ4 and Zlib libraries).

Testing
-------

After compiling, you can quickly check that the package is sane by
running::

  $ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
  $ export PYTHONPATH    (not needed on Windows)
  $ python -c"import bcolz; bcolz.test()"  # add `heavy=True` if desired

Installing
----------

Install it as a typical Python package::

  $ pip install -U .

Optionally Install the additional dependencies::

  $ pip install .[optional]

Documentation
-------------

You can find the online manual at:

http://bcolz.blosc.org

but of course, you can always access docstrings from the console
(i.e. ``help(bcolz.ctable)``).

Also, you may want to look at the bench/ directory for some examples
of use.

Resources
---------

Visit the main bcolz site repository at:
http://github.com/Blosc/bcolz

Home of Blosc compressor:
http://blosc.org

User's mail list:
http://groups.google.com/group/bcolz (bcolz@googlegroups.com)

An `introductory talk (20 min)
<https://www.youtube.com/watch?v=-lKV4zC1gss>`_ about bcolz at
EuroPython 2014.  `Slides here
<http://blosc.org/docs/bcolz-EuroPython-2014.pdf>`_.

License
-------

Please see ``BCOLZ.txt`` in ``LICENSES/`` directory.

Share your experience
---------------------

Let us know of any bugs, suggestions, gripes, kudos, etc. you may
have.

**Enjoy Data!**
