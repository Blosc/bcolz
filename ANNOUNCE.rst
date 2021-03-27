======================
Announcing bcolz 1.2.1
======================
cd
What's new
==========

This is a maintenance release where C-Blosc internal sources has been updated
to 1.14.3, in which Zstd codec should exhibit improved performance.  Also,
`np.datetime64` and other scalar objects that have `__getitem__()` are now
supported in _eval_blocks() (thanks to apalepu23). Finally, there is improved
support for ARM (specially for aarch64) and PowerPC architectures
(only little-endian).

For a more detailed change log, see:

https://github.com/Blosc/bcolz/blob/master/RELEASE_NOTES.rst

For some comparison between bcolz and other compressed data containers,
see:

https://github.com/FrancescAlted/DataContainersTutorials

specially chapters 3 (in-memory containers) and 4 (on-disk containers).


What it is
==========

*bcolz* provides **columnar and compressed** data containers that can
live either on-disk or in-memory.  The compression is carried out
transparently by Blosc, an ultra fast meta-compressor that is optimized
for binary data.  Compression is active by default.

Column storage allows for efficiently querying tables with a large
number of columns.  It also allows for cheap addition and removal of
columns.  Lastly, high-performance iterators (like ``iter()``,
``where()``) for querying the objects are provided.

bcolz can use diffent backends internally (currently numexpr,
Python/NumPy or dask) so as to accelerate many vector and query
operations (although it can use pure NumPy for doing so too).  Moreover,
since the carray/ctable containers can be disk-based, it is possible to
use them for seamlessly performing out-of-memory computations.

While NumPy is used as the standard way to feed and retrieve data from
bcolz internal containers, but it also comes with support for
high-performance import/export facilities to/from `HDF5/PyTables tables
<http://www.pytables.org>`_ and `pandas dataframes
<http://pandas.pydata.org>`_.

Have a look at how bcolz and the Blosc compressor, are making a better
use of the memory without an important overhead, at least for some real
scenarios:

http://nbviewer.ipython.org/github/Blosc/movielens-bench/blob/master/querying-ep14.ipynb#Plots

bcolz has minimal dependencies (NumPy is the only strict requisite),
comes with an exhaustive test suite, and it is meant to be used in
production. Example users of bcolz are Visualfabriq
(http://www.visualfabriq.com/), Quantopian (https://www.quantopian.com/)
and scikit-allel:

* Visualfabriq:

  * *bquery*, A query and aggregation framework for Bcolz:
  * https://github.com/visualfabriq/bquery

* Quantopian:

  * Using compressed data containers for faster backtesting at scale:
  * https://quantopian.github.io/talks/NeedForSpeed/slides.html

* scikit-allel:

  * Exploratory analysis of large scale genetic variation data.
  * https://github.com/cggh/scikit-allel


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

Release notes can be found in the Git repository:
https://github.com/Blosc/bcolz/blob/master/RELEASE_NOTES.rst

----

  **Enjoy data!**


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 72
.. End:
.. vim: set textwidth=72:
