==========================
Announcing bcolz 1.0.0 RC2
==========================

What's new
==========

Yeah, 1.0.0 is finally here.  We are not introducing any exciting new
feature (just some optimizations and bug fixes), but bcolz is already 6
years old and it implements most of the capabilities that it was
designed for, so I decided to release a 1.0.0 meaning that the format is
declared stable and that people can be assured that future bcolz
releases will be able to read bcolz 1.0 data files (and probably much
earlier ones too) for a long while.  Such a format is fully described
at:

https://github.com/Blosc/bcolz/blob/master/DISK_FORMAT_v1.rst

Also, a 1.0.0 release means that bcolz 1.x series will be based on
C-Blosc 1.x series (https://github.com/Blosc/c-blosc).  After C-Blosc
2.x (https://github.com/Blosc/c-blosc2) would be out, a new bcolz 2.x is
expected taking advantage of shiny new features of C-Blosc2 (more
compressors, more filters, native variable length support and the
concept of super-chunks), which should be very beneficial for next bcolz
generation.

Important: this is a Release Candidate, so please test it as much as you
can.  If no issues would appear in a week or so, I will proceed to tag
and release 1.0.0 final.  Enjoy!

For a more detailed change log, see:

https://github.com/Blosc/bcolz/blob/master/RELEASE_NOTES.rst


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
Blaze project (http://blaze.pydata.org/), Quantopian
(https://www.quantopian.com/) and Scikit-Allel
(https://github.com/cggh/scikit-allel) which you can read more about by
pointing your browser at the links below.

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

* Scikit-Allel

  * Provides an alternative backend to work with compressed arrays
  * https://scikit-allel.readthedocs.org/en/latest/model/bcolz.html

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
