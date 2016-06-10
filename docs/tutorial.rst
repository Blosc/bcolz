---------
Tutorials
---------

This section has been moved to ipython notebook `tutorials`_.

.. _tutorials: https://github.com/Blosc/bcolz/blob/master/docs/tutorials.ipynb

Tutorial on carray objects
==========================

This section has been moved to ipython notebook `tutorial_carray`_.

.. _tutorial_carray: https://github.com/Blosc/bcolz/blob/master/docs/tutorial_carray.ipynb

Tutorial on ctable objects
==========================

This section has been moved to ipython notebook `tutorial_ctable`_.

.. _tutorial_ctable: https://github.com/Blosc/bcolz/blob/master/docs/tutorial_ctable.ipynb

Writing bcolz extensions
========================

Did you like bcolz but you couldn't find exactly the functionality you were
looking for? You can write an extension and implement complex operations on
top of bcolz containers.

Before you start writing your own extension, let's see some
examples of real projects made on top of bcolz:

- `Bquery`: a query and aggregation framework, among other things it
   provides group-by functionality for bcolz containers. See
   https://github.com/visualfabriq/bquery

- `Bdot`: provides big dot products (by making your RAM bigger on the
   inside).  Supports ``matrix . vector`` and ``matrix . matrix`` for
   most common numpy numeric data types. See
   https://github.com/tailwind/bdot

Though not a extension itself, it is worth mentioning `Dask`. Dask
plays nicely with bcolz and provides multi-core execution on
larger-than-memory datasets using blocked algorithms and task
scheduling. See https://github.com/dask/dask.

In addition, bcolz also interacts well with `itertools`, `Pytoolz` or
`Cytoolz` too and they might offer you already the amount of
performance and functionality you are after.

In the next section we will go through all the steps needed to write
your own extension on top of bcolz.

How to use bcolz as part of the infrastructure
----------------------------------------------

Go to the root directory of bcolz, inside ``docs/my_package/`` you will
find a small extension example.

Before you can run this example you will need to install the following
packages.  Run ``pip install cython``, ``pip install numpy`` and ``pip
install bcolz`` to install these packages.  In case you prefer Conda
package management system execute ``conda install cython numpy bcolz``
and you should be ready to go.  See ``requirements.txt``:

.. literalinclude:: my_package/requirements.txt
    :language: python

Once you have those packages installed, change your working directory
to ``docs/my_package/``, please see `pkg. example
<https://github.com/Blosc/bcolz/tree/master/docs/my_package>`_ and run
``python setup.py build_ext --inplace`` from the terminal, if
everything ran smoothly you should be able to see a binary file
``my_extension/example_ext.so`` next to the ``.pyx`` file.

If you have any problems compiling these extensions, please make sure
you have a recent version of bcolz as old versions (pre 0.8) don't
contain the necessary ``.pxd`` file which provides a Cython interface
to the carray Cython module.

The ``setup.py`` file is where you will need to tell the compiler, the
name of you package, the location of external libraries (in case you
want to use them), compiler directives and so on.  See `bcolz setup.py
<https://github.com/Blosc/bcolz/blob/master/setup.py>`_ as a possible
reference for a more complete example.  Along your project grows in
complexity you might be interested in including other options to your
`Extension` object, e.g. `include_dirs` to include a list of
directories to search for C/C++ header files your code might be
dependent on.

See ``my_package/setup.py``:

.. literalinclude:: my_package/setup.py
    :language: python

The ``.pyx`` files is going to be the place where Cython code
implementing the extension will be, in the example below the function
will return a sum of all integers inside the carray.

See ``my_package/my_extension/example_ext.pyx``

Keep in mind that carrays are great for sequential access, but random
access will highly likely trigger decompression of a different chunk
for each randomly accessed value.

For more information about Cython visit http://docs.cython.org/index.html

.. literalinclude:: my_package/my_extension/example_ext.pyx
    :language: python

Let's test our extension:

        >>> import bcolz
        >>> import my_extension.example_ext as my_mod
        >>> c = bcolz.carray([i for i in range(1000)], dtype='i8')
        >>> my_mod.my_function(c)
        499500
