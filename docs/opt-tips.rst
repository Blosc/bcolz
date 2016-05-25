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
    cparams := cparams(clevel=5, shuffle=1)
  [0.0, 1.0, 2.0, ..., 9999997.0, 9999998.0, 9999999.0]
  >>> bcolz.carray(a).chunklen
  16384   # 128 KB = 16384 * 8 is the default chunk size for this carray
  >>> bcolz.carray(a, chunklen=512)
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 10.20 MB; ratio: 7.48
    cparams := cparams(clevel=5, shuffle=1)
  [0.0, 1.0, 2.0, ..., 9999997.0, 9999998.0, 9999999.0]
  >>> bcolz.carray(a, chunklen=8*1024)
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 1.50 MB; ratio: 50.88
    cparams := cparams(clevel=5, shuffle=1)
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

Lossy compression via the quantize filter
=========================================

Using the `quantize` filter for allowing lossy compression on floating
point data.  Data is quantized using ``np.around(scale*data)/scale``,
where scale is 2**bits, and bits is determined from the quantize
value.  For example, if quantize=1, bits will be 4.  0 means that the
quantization is disabled.

Here it is an example of what you can get from the quantize filter::

  In [9]: a = np.cumsum(np.random.random_sample(1000*1000)-0.5)

  In [10]: bcolz.carray(a, cparams=bcolz.cparams(quantize=0))  # no quantize
  Out[10]:
  carray((1000000,), float64)
    nbytes: 7.63 MB; cbytes: 6.05 MB; ratio: 1.26
    cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=0)
  [ -2.80946077e-01  -7.63925274e-01  -5.65575047e-01 ...,   3.59036158e+02
     3.58546624e+02   3.58258860e+02]

  In [11]: bcolz.carray(a, cparams=bcolz.cparams(quantize=1))
  Out[11]:
  carray((1000000,), float64)
    nbytes: 7.63 MB; cbytes: 1.41 MB; ratio: 5.40
    cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=1)
  [ -2.50000000e-01  -7.50000000e-01  -5.62500000e-01 ...,   3.59036158e+02
     3.58546624e+02   3.58258860e+02]

  In [12]: bcolz.carray(a, cparams=bcolz.cparams(quantize=2))
  Out[12]:
  carray((1000000,), float64)
    nbytes: 7.63 MB; cbytes: 2.20 MB; ratio: 3.47
    cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=2)
  [ -2.81250000e-01  -7.65625000e-01  -5.62500000e-01 ...,   3.59036158e+02
     3.58546624e+02   3.58258860e+02]

  In [13]: bcolz.carray(a, cparams=bcolz.cparams(quantize=3))
  Out[13]:
  carray((1000000,), float64)
    nbytes: 7.63 MB; cbytes: 2.30 MB; ratio: 3.31
    cparams := cparams(clevel=5, shuffle=1, cname='blosclz', quantize=3)
  [ -2.81250000e-01  -7.63671875e-01  -5.65429688e-01 ...,   3.59036158e+02
     3.58546624e+02   3.58258860e+02]

As you can see, the compression ratio can improve pretty significantly
when using the quantize filter.  It is important to note that by using
quantize you are loosing precision on your floating point data.

Also note how the first elements in the quantized arrays have less
significant digits, but not the last ones.  This is a side effect due
to how bcolz stores the trainling data that do not fit in a whole
chunk.  But in general you should expect a loss in precision.
