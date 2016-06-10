======================
Announcing bcolz 1.1.0
======================

What's new
==========

This release brings quite a lot of changes.  After format stabilization
in 1.0, the focus is now in fine-tune many operations (specially queries
in ctables), as well as widening the available computational engines.

Highlights:

* Much improved performance of ctable.where() and ctable.whereblocks().
  Now bcolz is getting closer than ever to fundamental memory limits
  during queries (see the updated benchmarks in the data containers
  tutorial below).

* Better support for Dask; i.e. GIL is released during Blosc operation
  when bcolz is called from a multithreaded app (like Dask).  Also, Dask
  can be used as another virtual machine for evaluating expressions (so
  now it is possible to use it during queries too).

* New ctable.fetchwhere() method for getting the rows fulfilling some
  condition in one go.

* New quantize filter for allowing lossy compression of floating point
  data.

* It is possible to create ctables with more than 255 columns now.
  Thanks to Skipper Seabold.

* The defaults during carray creation are scalars now.  That allows to
  create highly dimensional data containers more efficiently.

* carray object does implement the __array__() special method now. With
  this, interoperability with numpy arrays is easier and faster.

For a more detailed change log, see:

https://github.com/Blosc/bcolz/blob/master/RELEASE_NOTES.rst

For some comparison between bcolz and other compressed data containers,
see:

https://github.com/FrancescAlted/DataContainersTutorials

specially chapters 3 (in-memory containers) and 4 (on-disk containers).


What it is
==========

*bcolz* provides columnar and compressed data containers that can live
either on-disk or in-memory.  Column storage allows for efficiently
querying tables with a large number of columns.  It also allows for
cheap addition and removal of column.  In addition, bcolz objects are
compressed by default for reducing memory/disk I/O needs. The
compression process is carried out internally by Blosc, an
extremely fast meta-compressor that is optimized for binary data. Lastly,
high-performance iterators (like ``iter()``, ``where()``) for querying
the objects are provided.

bcolz can use numexpr internally so as to accelerate many vector and
query operations (although it can use pure NumPy for doing so too).
numexpr optimizes the memory usage and use several cores for doing the
computations, so it is blazing fast.  Moreover, since the carray/ctable
containers can be disk-based, and it is possible to use them for
seamlessly performing out-of-memory computations.

bcolz has minimal dependencies (NumPy), comes with an exhaustive test
suite and fully supports both 32-bit and 64-bit platforms.  Also, it is
typically tested on both UNIX and Windows operating systems.

Together, bcolz and the Blosc compressor, are finally fulfilling the
promise of accelerating memory I/O, at least for some real scenarios:

http://nbviewer.ipython.org/github/Blosc/movielens-bench/blob/master/querying-ep14.ipynb#Plots

Example users of bcolz are Visualfabriq (http://www.visualfabriq.com/),
and Quantopian (https://www.quantopian.com/):

* Visualfabriq:

  * *bquery*, A query and aggregation framework for Bcolz:
  * https://github.com/visualfabriq/bquery

* Quantopian:

  * Using compressed data containers for faster backtesting at scale:
  * https://quantopian.github.io/talks/NeedForSpeed/slides.html



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
