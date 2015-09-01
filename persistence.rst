=====================================
RFC for a persistence layer for bcolz
=====================================

:Author: Francesc Alted
:Contact: francesc@blosc.org
:Version: 0.1 (August 19, 2012)


    The original bcolz container (up to version 0.4) consisted on
    basically a list of compressed in-memory blocks.  This document
    explains how to extend it to allow to store the data blocks on disk
    too.

    The goals of this proposal are:

    1. Allow to work with data directly on disk, exactly on the same way
      than data in memory.

    2. Must support the same access capabilities than bcolz objects
       including: append data, modifying data and direct access to data.

    3. Transparent data compression must be possible.

    4. User metadata addition must be possible too.

    5. The data should be easily 'shardeable' for optimal behaviour in a
       distributed storage environment.

    This, in combination with a distributed filesystem, and combined with
    a system that is aware of the physical topology of the
    underlying storage media would allow to almost replace the need for
    a distributed infrastructure for data (e.g. Disco/Hadoop).

The layout
==========

For every dataset, it will be created a directory, with a
user-provided name that, for generality, we will call it `root` here.
The root will have another couple of subdirectories, named data and
meta::

        root  (the name of the dataset)
        /  \
     data  meta

The `data` directory will contain the actual data of the dataset,
while the `meta` will contain the metainformation (dtype, shape,
chunkshape, compression level, filters...).

The `data` layout
-----------------

Data will be stored by what is called a `superchunk`, and each
superchunk will use exactly one file.  The size of each superchunk
will be decided automatically by default, but it could be specified by
the user too.

Each of these directories will contain one or more superchunks for
storing the actual data.  Every data superchunk will be named after
its sequential number.  For example::

    $ ls data
    __1__.bin  __2__.bin  __3__.bin  __4__.bin ... __1030__.bin

This structure of separate superchunk files allows for two things:

1. Datasets can be enlarged and shrinked very easily
2. Horizontal sharding in a distributed system is possible (and cheap!)

At its time, the `data` directory might contain other subdirectories
that are meant for storing components for a 'nested' dtype (i.e. an
structured array, stored in column-wise order)::

        data  (the root for a nested datatype)
        /  \     \
     col1  col2  col3
          /  \
        sc1  sc3

This structure allows for quick access to specific chunks of columns
without a need to load the complete data in memory.

The `superchunk` layout
~~~~~~~~~~~~~~~~~~~~~~~

The superchunk is made of a series of data chunks put together using
the Blosc metacompressor by default.  Blosc being a metacompressor,
means that it can use different compressors and filters, while
leveraging its blocking and multithreading capabilities.

The layout of binary superchunk data files looks like this::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    | b   l   p   k | ^ | ^ | ^ | ^ |   chunk-size  |  last-chunk   |
                      |   |   |   |
          version ----+   |   |   |
          options --------+   |   |
         checksum ------------+   |
         typesize ----------------+

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
    |            nchunks            |            RESERVED           |


The magic 'blpk' signature is the same than the bloscpack_ format.
The new version (2) of the format will allow to include indexes
(offsets to where the data chunks begin) and checksums (probably using
the adler32 algorithm or similar).

.. _blosckpack: https://github.com/esc/bloscpack/blob/feature/new_format/header_rfc.rst

After the above header, it will follow index data and the actual data
in blosc chunks::

    |-bloscpack-header-|-offset-|-offset-|...|-chunk-|-chunk-|...|

The index part above stores the offsets where each chunk starts, so it
is is easy to access the different chunks in the superchunk file.

CAVEAT: The bloscpack format is still evolving, so don't trust on
forward compatibility of the format, at least until 1.0, where the
internal format will be declared frozen.

And each blosc chunk has this format (Blosc 1.0 on)::

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------blosclz version
      +--------------blosc version

At the end of each blosc chunk some empty space could be added in
order to allow the modification of some data elements inside each
block.  The reason for the additional space is that, as these chunks
will be typically compressed, when modifying some element of the chunk
it is not guaranteed that it will fit in the same space than the old
data chunk.  Having this provision of small empty space at the end of
each chunk will allow for storing the modifyed chunks in many cases,
without a need to save the entire superchunk on a different part of
the disk.

The `meta` files
----------------

Here there can be as many files as necessary.  The format for every
file will tentatively be YAML (although initial implementations are
using JSON).  There should be (at least) three files:

The `sizes` file
~~~~~~~~~~~~~~~~

This contains the shape and compressed and uncompressed sizes of the
dataset.  For example::

    $ cat meta/sizes
    shape: (5000000000,)
    nbytes: 5000000000
    cbytes: 24328038

The `storage` file
~~~~~~~~~~~~~~~~~~

Here comes the information about how data has to be stored and its
meaning. Example::

    dtype: 
      col1: int8
      col2: float32
    chunkshape: (30, 20)
    superchunksize: 10  # max. number of chunks in a single file
    endianness: big  # default: little
    order: C         # default: C
    compression:
      library: blosclz   # could be zlib, fastlz or others
      level: 5
      filters: [shuffle, truncate]  # order matters

The `attributes` file
~~~~~~~~~~~~~~~~~~~~~

In this file it comes additional user information.  Example::

    temperature:
      value: 23.5
      type: scalar
      dtype: float32
    pressure:
      value: 225.5
      type: scalar
      dtype: float32
    ids:
      value: [1,3,6,10]
      type: array
      dtype: int32

More files could be added for providing other kind of meta-information
about data (read indexes, masks...).
