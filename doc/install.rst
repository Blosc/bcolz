------------
Installation
------------

carray depends on NumPy (>= 1.4.1) and, optionally, Numexpr (>= 1.4).
Also, if you are going to install from sources, and a C compiler (both
GCC and MSVC 2008 have been tested).

Installing from sources
=======================

Go to the carray main directory and do the typical distutils dance::

  $ python setup.py build_ext -i
  $ export PYTHONPATH=.   # set PYTHONPATH=.  on Windows
  $ python carray/tests/test_all.py
  $ python setup.py install

Installing from Windows binaries
================================

Just download the binary installer and run it.


