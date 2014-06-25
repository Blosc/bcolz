------------
Installation
------------

bcolz depends on NumPy (>= 1.5) and, optionally, Numexpr (>= 1.4).
Also, if you are going to install from sources, and a C compiler (both
GCC and MSVC 2008 have been tested).

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

  $ python setup.py build_ext -i
  $ export PYTHONPATH=.   # set PYTHONPATH=.  on Windows
  $ python bcolz/tests/test_all.py
  $ python setup.py install


Installing from the git repository
==================================

If you have cloned the bcolz repository, you can follow the same
procedure than for the tarball above, but you may also want to use Paver
(http://paver.github.com/paver/) for compiling and generating docs.
So, first install Paver and then::

  $ paver build_ext -i
  $ export PYTHONPATH=.   # set PYTHONPATH=.  on Windows
  $ python bcolz/tests/test_all.py
  $ paver install

Also, you can generate documentation in both pdf and html formats::

  $ paver pdf      # PDF output in doc/bcolz-manual.pdf
  $ paver html     # HTML output in doc/html/


Testing the installation
========================

You can always test the installation from any directory with::

  $ python -c "import bcolz; bcolz.test()"


