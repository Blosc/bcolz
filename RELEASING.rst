===============
Releasing bcolz
===============

:Author: Francesc Alted
:Contact: francesc@blosc.org
:Date: 2014-07-20


Preliminaries
-------------

* Make sure that ``RELEASE_NOTES.rst`` and ``ANNOUNCE.rst`` are up to
  date with the latest news in the release.

* Commit your changes::

    $ git commit -a -m"Getting ready for X.Y.Z final"

* Once a year: check that the copyright year in ``LICENSES/BCOLZ.txt``
  and in ``docs/conf.py`` is up to date.


Tagging
-------

* Create a tag ``X.Y.Z`` from ``master``.  Use the next message::

    $ git tag -a X.Y.Z -m "Tagging version X.Y.Z"

  Note: For release candidates, just add a rcN suffix to tag ("X.Y.ZrcN").

* Or, alternatively, make a signed tag (requires gpg correctly configured)::

    $ git tag -s X.Y.Z -m "Tagging version X.Y.Z"

* Push the tag to the Github repo::

    $ git push
    $ git push --tags


Testing
-------

* After compiling, run::

  $ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
  $ export PYTHONPATH=.  (not needed on Win)
  $ python -c "import bcolz; bcolz.test(heavy=True)"

* Run the test suite in different platforms (at least Linux and
  Windows) and make sure that all tests passes.


Packaging
---------

* Make sure that you are in a clean directory.  The best way is to
  re-clone and re-build::

  $ cd /tmp
  $ git clone git@github.com:Blosc/bcolz.git
  $ cd bcolz
  $ python setup.py build_ext

* Check that all Cython generated ``*.c`` files are present.

* Make the tarball with the command::

  $ python setup.py sdist

Do a quick check that the tarball is sane.


Uploading
---------

* Upload it also in the PyPi repository::

    $ python setup.py sdist upload


Announcing
----------

* Send an announcement to the bcolz, blosc, pydata and python-announce
  lists.  Use the ``ANNOUNCE.rst`` file as skeleton (or possibly as
  the definitive version).

* Tweet about the new release and rejoice!


Post-release actions
--------------------

* Create new headers for adding new features in ``RELEASE_NOTES.rst``
  and add this place-holder instead:

  #XXX version-specific blurb XXX#

* Commit your changes with:

  $ git commit -a -m"Post X.Y.Z release actions done"


That's all folks!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
