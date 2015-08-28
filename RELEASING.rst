===============
Releasing bcolz
===============

:Author: Francesc Alted
:Contact: faltet@blosc.io
:Date: 2014-07-20


Preliminaries
-------------

* Make sure that ``RELEASE_NOTES.rst`` and ``ANNOUNCE.rst`` are up to
  date with the latest news in the release.

* Tag the version.

* Once a year: check that the copyright in ``LICENSES/BCOLZ.txt`` and
  ``doc/conf.py``.

Tagging
-------

* Create a tag ``X.Y.Z`` from ``master``.  Use the next message::

    $ git tag -a X.Y.Z -m "Tagging version X.Y.Z"

* Or, alternatively, make a signed tag (requires gpg correctly configured)::

    $ git tag -s X.Y.Z -m "Tagging version X.Y.Z"

* Push the tag to the Github repo (assuming ``origin`` is correct)::

    $ git push origin X.Y.Z



Testing
-------

* After compiling, run::

  $ PYTHONPATH=.   (or "set PYTHONPATH=." on Win)
  $ export PYTHONPATH=.  (not needed on Win)
  $ python -c "import bcolz; bcolz.test(heavy=True)"

* Run the test suite in different platforms (at least Linux and
  Windows) and make sure that all tests passes.


Updating the online documentation site
--------------------------------------

.. note::

    This instructions are currently out-of-date and are to be considered under
    construction.

* Build the html::

  $ python setup.py build_sphinx

* Make a backup and upload the files in the doc site (xodo)::

  $ export UPSTREAM="/home/blosc/srv/www/bcolz.blosc.org"
  $ ssh blosc@xodo.blosc.org "mv $UPSTREAM/docs/html $UPSTREAM/docs/html.bck"
  $ scp -r build/sphinx/html blosc@xodo.blosc.org:$UPSTREAM/docs

* Check that the new manual is accessible in http://bcolz.blosc.org

* If everything is fine, remove the backup of the previous manual::

  $ ssh blosc@xodo.blosc.org "rm -r $UPSTREAM/docs/html.bck"

* Go up to the root directory for further proceeding with packaging::

  $ cd ..


Packaging
---------

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

* Send an announcement to the bcolz, blosc, numpy, pandas and
  python-announce lists.  Use the ``ANNOUNCE.rst`` file as skeleton
  (or possibly as the definitive version).

* Tweet about the new release and enjoy!


Post-release actions
--------------------

* Create new headers for adding new features in ``RELEASE_NOTES.rst``
  and empty the release-specific information in ``ANNOUNCE.rst`` and
  add this place-holder instead:

  #XXX version-specific blurb XXX#


That's all folks!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
