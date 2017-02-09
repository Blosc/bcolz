------------
Installation
------------

bcolz depends on NumPy and, optionally, Numexpr.  Also, if you are
going to install from sources, and a C compiler (Clang, GCC and MSVC
2008 for Python 2, and MSVC 2010 for Python 3, have been tested).


Installing from PyPI repository
===============================

Do::

  $ easy_install -U bcolz

or::

  $ pip install -U bcolz

Installing from conda-forge
===========================

Binaries for Linux, Mac and Windows are available for installation via conda. 
Do::

  $ conda install -c conda-forge bcolz

Installing Windows binaries
===========================

Unofficial Windows binaries are provided by Christoph Gohlke and can be
downloaded from:

http://www.lfd.uci.edu/~gohlke/pythonlibs/#bcolz

Using the Microsoft Python 2.7 Compiler
=======================================

As of Sept 2014 Microsoft has made a Visual C++ compiler for Python 2.7
available for download:

http://aka.ms/vcpython27

This has been made available specifically to ease the handling of Python
packages with C-extensions on Windows (installation and building wheels).

It is possible to compile bcolz with this compiler (Jan 2015), however,
you may need to use the following patch::

    diff --git i/setup.py w/setup.py
    index d77d37f233..b54bfd0fa1 100644
    --- i/setup.py
    +++ w/setup.py
    @@ -11,8 +11,8 @@ from __future__ import absolute_import
     import sys
     import os
     import glob
    -from distutils.core import Extension
    -from distutils.core import setup
    +from setuptools import Extension
    +from setuptools import setup
     import textwrap
     import re, platform

Installing from tarball sources
===============================

Go to the bcolz main directory and do the typical distutils dance::

    $ python setup.py build_ext --inplace

In case you have Blosc installed as an external library you can link
with it (disregarding the included Blosc sources) in a couple of ways:

Using an environment variable::

  $ BLOSC_DIR=/usr/local     (or "set BLOSC_DIR=\blosc" on Win)
  $ export BLOSC_DIR         (not needed on Win)
  $ python setup.py build_ext --inplace

Using a flag::

  $ python setup.py build_ext --inplace --blosc=/usr/local

It is always nice to run the tests before installing the package::

  $ PYTHONPATH=.   (or "set PYTHONPATH=." on Windows)
  $ export PYTHONPATH    (not needed on Windows)
  $ python -c"import bcolz; bcolz.test()"  # add `heavy=True` if desired

And if everything runs fine, then install it via::

  $ python setup.py install


Testing the installation
========================

You can always test the installation from any directory with::

  $ python -c "import bcolz; bcolz.test()"

