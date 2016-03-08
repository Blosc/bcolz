==========================================
The persistence layer for bcolz 1.x series
==========================================

:Author: Francesc Alted
:Contact: francesc@blosc.org
:Version: 1.0 (March 8, 2016)


This document explains how the data is stored in the bcolz format.
The goals of this format are:

1. Allow to work with datasets (carray/ctable) directly on disk,
   exactly on the same way than data in memory.

2. Must support the same access capabilities than carray/ctable
   objects including: append data, modying data and direct access to
   data.

3. Transparent data compression must be possible.

4. The data should be easily 'shardeable' for optimal behaviour in a
   distributed storage environment.

5. User metadata addition must be possible too.


The layout
==========

For every dataset, a directory is created, with a user-provided name
that, for generality, we will call it `root` here. The root will have
another couple of subdirectories, named data and meta::

        root  (the name of the dataset)
        /  \
     data  meta

The `data` directory contains the actual data of the dataset, while
the `meta` will contain the meta-information (dtype, shape,
chunkshape, compression level, filters...).


The `data` layout
-----------------

Data is stored in data blocks that are called `superchunks`, and each
superchunk will use exactly one file.  The size of each superchunk
will be decided automatically by default, but it could be specified by
the user too.

Each of these directories will contain one or more chunks for storing
the actual data.  Every data chunk will be named after its sequential
number.  For example::

    $ ls data
    __0.blp  __1.blp  __2.blp  __3.blp ... __1030.blp

This structure of separate superchunk files allows for two things:

1. Datasets can be enlarged and shrinked very easily.

2. Horizontal sharding in a distributed system is possible (and cheap!).

At its time, the `data` directory might contain other subdirectories
that are meant for storing components for a ctable (i.e. an structured
array like, but stored in column-wise order)::

        data  (the root for a nested datatype)
        /  \     \
     col1  col2  col3  (first-level colmuns)
          /  \
        sc1  sc3    (-> nested columns, if exist)

This structure allows for quick access to specific superchunks of
columns without a need to load the complete dataset in memory.

The `superchunk` layout
~~~~~~~~~~~~~~~~~~~~~~~

The superchunk is made of a series of data blocks put together using
the C-Blosc 1.x metacompressor by default.  Blosc being a
metacompressor, means that it can use different compressors and
filters, while leveraging its blocking and multithreading
capabilities.

The layout of binary superchunk data files looks like this::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | RESERVED  |           nchunks             |
                   version

The first four are the magic string 'blpk'. The next one is an 8 bit
unsigned little-endian integer that encodes the format version. The
next three are reserved, and in the last eight there is a signed 64
bit little endian integer that encodes the number of Blosc chunks
inside the superchunk.

Currently (bcolz 1.x), version is 1 and nchunks always has a
value of 1 (this might change in bcolz 2.0).

After the above header, it follows the actual data in Blosc chunk.  At
its time, each chunk has this format (Blosc 1.x)::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------blosclz version
      +--------------blosc version

For more details on this, see the `C-Blosc header description
<https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst>`_.

The `meta` files
----------------

Here there can be as many files as necessary.  The format for every
file is JSON.  There should be (at least) two files:

The `sizes` file
~~~~~~~~~~~~~~~~

This contains the shape and compressed and uncompressed sizes of the
dataset.  For example::

    $ cat meta/sizes
    {"shape": [100000], "nbytes": 400000, "cbytes": 266904}

The `storage` file
~~~~~~~~~~~~~~~~~~

Here comes the information about how data has to be stored and its
meaning. Example::

    $ cat meta/sizes
    {"dtype": "int32", "cparams": {"shuffle": true, "clevel": 5}, "chunklen": 65536, "dflt": 0, "expectedlen": 100000}

The `__attrs__` file
---------------------

Finally, in this file (placed at the root directory for each dataset)
it comes additional user information (not mandatory) serialized in
JSON format.  Example::

  $ cat __attrs__
  {"temp": 22.5, "pressure": 999.2, "timestamp": "2016030915"}
