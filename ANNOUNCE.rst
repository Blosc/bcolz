======================
Announcing bcolz 0.8.0
======================

What's new
==========

This version adds a public API in the form of a Cython definitions file
(``carray_ext.pxd``) for the ``carray`` class!

This means, other libraries can use the Cython definitions to build more
complex programs using the objects provided by bcolz. In fact, this
feature was specifically requested and there already exists a nascent
application called *bquery* (https://github.com/visualfabriq/bquery)
which provides an efficient out-of-core groupby implementation for the
``ctable`` object

Because this is a fairly sweeping change the minor version number was
incremented and no additional major features or bugfixes were added to
this release.  We kindly ask any users of bcolz to try this version
carefully and report back any issues, bugs, or even slow-downs you
experience.  I.e. please, please be careful when deploying this version
into production.

Many, many kudos to Francesc Elies and Carst Vaartjes of Visualfabriq
for their hard work, continued effort to push this feature and their
work on bquery which makes use of it!

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

Other users of bcolz are Visualfabriq (http://www.visualfabriq.com/) the
Blaze project (http://blaze.pydata.org/) and Quantopian
(https://www.quantopian.com/) which you can read more about by pointing
your browser at the links below.

* Visualfabriq:

  * *bquery*, A query and aggregation framework for Bcolz:
  * https://github.com/visualfabriq/bquery

* Blaze:

  * Notebooks showing Blaze + Pandas + BColz interaction: 
  * http://nbviewer.ipython.org/url/blaze.pydata.org/notebooks/timings-csv.ipynb
  * http://nbviewer.ipython.org/url/blaze.pydata.org/notebooks/timings-bcolz.ipynb

* Quantopian:

  * Using compressed data containers for faster backtesting at scale:
  * https://quantopian.github.io/talks/NeedForSpeed/slides.html

Installing
==========

bcolz is in the PyPI repository, so installing it is easy::

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
