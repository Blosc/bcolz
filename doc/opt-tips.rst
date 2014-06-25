.. _opt-tips:

-----------------
Optimization tips
-----------------

Changing explicitly the length of chunks
========================================

You may want to use explicitly the `chunklen` parameter to fine-tune
your compression levels::

  >>> a = np.arange(1e7)
  >>> bcolz.carray(a)
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 2.57 MB; ratio: 29.72
    cparams := cparams(clevel=5, shuffle=True)
  [0.0, 1.0, 2.0, ..., 9999997.0, 9999998.0, 9999999.0]
  >>> bcolz.carray(a).chunklen
  16384   # 128 KB = 16384 * 8 is the default chunk size for this carray
  >>> bcolz.carray(a, chunklen=512)
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 10.20 MB; ratio: 7.48
    cparams := cparams(clevel=5, shuffle=True)
  [0.0, 1.0, 2.0, ..., 9999997.0, 9999998.0, 9999999.0]
  >>> bcolz.carray(a, chunklen=8*1024)
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 1.50 MB; ratio: 50.88
    cparams := cparams(clevel=5, shuffle=True)
  [0.0, 1.0, 2.0, ..., 9999997.0, 9999998.0, 9999999.0]

You see, the length of the chunk affects very much compression levels
and the performance of I/O to carrays too.

In general, however, it is safer (and quicker!) to use the
`expectedlen` parameter (see next section).

Informing about the length of your carrays
==========================================

If you are going to add a lot of rows to your carrays, be sure to use
the `expectedlen` parameter in creating time to inform the constructor
about the expected length of your final carray; this allows bcolz to
fine-tune the length of its chunks more easily.  For example::

  >>> a = np.arange(1e7)
  >>> bcolz.carray(a, expectedlen=10).chunklen
  512
  >>> bcolz.carray(a, expectedlen=10*1000).chunklen
  4096
  >>> bcolz.carray(a, expectedlen=10*1000*1000).chunklen
  16384
  >>> bcolz.carray(a, expectedlen=10*1000*1000*1000).chunklen
  131072
