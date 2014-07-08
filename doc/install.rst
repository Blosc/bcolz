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


Installing from Windows binaries
================================

Just download the binary installer and run it.


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


Installing from the git repository
==================================

If you have cloned the bcolz repository, you can follow the same
procedure than for the tarball above, but you may also want to use Paver
(http://paver.github.com/paver/) for compiling and generating docs.
So, first install Paver and then::

  $ paver build_ext -i
  $ export PYTHONPATH=.   # set PYTHONPATH=.  on Windows
  $ python -c"import bcolz; bcolz.test()"  # add `heavy=True` if desired
  $ paver install

Also, you can generate documentation in both pdf and html formats::

  $ paver pdf      # PDF output in doc/bcolz-manual.pdf
  $ paver html     # HTML output in doc/html/


Testing the installation
========================

You can always test the installation from any directory with::

  $ python -c "import bcolz; bcolz.test()"

