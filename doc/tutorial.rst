---------
Tutorials
---------

Tutorial on carray objects
==========================

Creating carrays
----------------

A carray can be created from any NumPy ndarray by using its `carray`
constructor::

  >>> a = np.arange(10)
  >>> b = bcolz.carray(a)                          # for in-memory storage
  >>> with bcolz.carray(a, rootdir='mydir') as _:  # for on-disk storage
  ...     c = _  # keep the reference to the Python object

To avoid forgetting to flush your data to disk, you are encouraged to use the
`with` statement for on-disk carrays.

Or, you can also create it by using one of its multiple constructors
(see :ref:`top-level-constructors` for the complete list), write mode will
overwrite contents of the folder where the carray is created::

  >>> with bcolz.arange(10, rootdir='mydir', mode='w') as _:
  ...     d = _

Please note that carray allows to create disk-based arrays by just
specifying the `rootdir` parameter in all the constructors.
Disk-based arrays fully support all the operations of in-memory
counterparts, so depending on your needs, you may want to use one or
another (or even a combination of both).

Now, `b` is a carray object.  Just check this::

  >>> type(b)
  <type 'carray.carrayExtension.carray'>

You can have a peek at it by using its string form::

  >>> print b
  [0, 1, 2... 7, 8, 9]

And get more info about uncompressed size (nbytes), compressed
(cbytes) and the compression ratio (ratio = nbytes/cbytes), by using
its representation form::

  >>> b   # <==> print repr(b)
  carray((10,), int64)
    nbytes: 80; cbytes: 16.00 KB; ratio: 0.00
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0 1 2 3 4 5 6 7 8 9]

As you can see, the compressed size is much larger than the
uncompressed one.  How this can be?  Well, it turns out that carray
wears an I/O buffer for accelerating some internal operations.  So,
for small arrays (typically those taking less than 1 MB), there is
little point in using a carray.

However, when creating carrays larger than 1 MB (its natural
scenario), the size of the I/O buffer is generally negligible in
comparison::

  >>> b = bcolz.arange(1e8)
  >>> b
  carray((100000000,), float64)
  nbytes: 762.94 MB; cbytes: 23.25 MB; ratio: 32.82
  cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999970e+07
     9.99999980e+07   9.99999990e+07]

The carray consumes less than 24 MB, while the original data would have
taken more than 760 MB; that's a huge gain.  You can always get a hint
on how much space it takes your carray by using `sys.getsizeof()`::

  >>> import sys
  >>> sys.getsizeof(b)
  24376698

That moral here is that you can create very large arrays without the
need to create a NumPy array first (that may not fit in memory).

Finally, you can get a copy of your created carrays by using the
`copy()` method::

  >>> c = b.copy()
  >>> c
  carray((100000000,), float64)
    nbytes: 762.94 MB; cbytes: 23.25 MB; ratio: 32.82
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999970e+07
     9.99999980e+07   9.99999990e+07]

and you can control parameters for the newly created copy::

  >>> b.copy(cparams=bcolz.cparams(clevel=9))
  carray((100000000,), float64)
    nbytes: 762.94 MB; cbytes: 8.09 MB; ratio: 94.27
    cparams := cparams(clevel=9, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999970e+07
     9.99999980e+07   9.99999990e+07]

Enlarging your carray
---------------------

One of the nicest features of carray objects is that they can be
enlarged very efficiently.  This can be done via the `carray.append()`
method.

For example, if `b` is a carray with 10 million elements::

  >>> b = bcolz.arange(10*1e6)
  >>> b
  carray((10000000,), float64)
    nbytes: 76.29 MB; cbytes: 2.94 MB; ratio: 25.92
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999700e+06
     9.99999800e+06   9.99999900e+06]

it can be enlarged by 10 elements with::

  >>> b.append(np.arange(10.))
  >>> b
  carray((10000010,), float64)
    nbytes: 76.29 MB; cbytes: 2.94 MB; ratio: 25.92
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [ 0.  1.  2. ...,  7.  8.  9.]

Let's check how fast appending can be::

  >>> a = np.arange(1e7)
  >>> b = bcolz.arange(1e7)
  >>> %time b.append(a)
  CPU times: user 51.2 ms, sys: 15.8 ms, total: 67 ms
  Wall time: 24.5 ms
  >>> %time np.concatenate((a, a))
  CPU times: user 44.4 ms, sys: 45 ms, total: 89.4 ms
  Wall time: 91 ms # 3.7x slower than carray 
  array([  0.00000000e+00,   1.00000000e+00,   2.00000000e+00, ...,
           9.99999700e+06,   9.99999800e+06,   9.99999900e+06])

This is specially true when appending small bits to large arrays::

  >>> b = bcolz.carray(a)
  >>> %timeit b.append(np.arange(1e1))
  100000 loops, best of 3: 3.24 µs per loop
  >>> %timeit np.concatenate((a, np.arange(1e1)))
  10 loops, best of 3: 25.2 ms per loop  # ~10000X slower than carray

You can also enlarge your arrays by using the `resize()` method::

  >>> b = bcolz.arange(10)
  >>> b.resize(20)
  >>> b
  carray((20,), int64)
    nbytes: 160; cbytes: 16.00 KB; ratio: 0.01
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0 1 2 3 4 5 6 7 8 9 0 0 0 0 0 0 0 0 0 0]

Note how the append values are filled with zeros.  This is because the
default value for filling is 0.  But you can choose a different value
too::

  >>> b = bcolz.arange(10, dflt=1)
  >>> b.resize(20)
  >>> b
  carray((20,), int64)
    nbytes: 160; cbytes: 16.00 KB; ratio: 0.01
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0 1 2 3 4 5 6 7 8 9 1 1 1 1 1 1 1 1 1 1]

Also, you can trim carrays::

  >>> b = bcolz.arange(10)
  >>> b.resize(5)
  >>> b
  carray((5,), int64)
    nbytes: 40; cbytes: 16.00 KB; ratio: 0.00
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0 1 2 3 4]

You can even set the size to 0:

  >>> b.resize(0)
  >>> len(b)
  0

Definitely, resizing is one of the strongest points of carray
objects, so do not be afraid to use that feature extensively.

Compression level and shuffle filter
------------------------------------

carray uses Blosc as the internal compressor, and Blosc can be
directed to use different compression levels and to use (or not) its
internal shuffle filter.  The shuffle filter is a way to improve
compression when using items that have type sizes > 1 byte, although
it might be counter-productive (very rarely) for some data
distributions.

By default carrays are compressed using Blosc with compression level 5
with shuffle active.  But depending on you needs, you can use other
compression levels too::

  >>> a = np.arange(1e7)
  >>> bcolz.carray(a, bcolz.cparams(clevel=1))
  carray((10000000,), float64)
    nbytes: 76.29 MB; cbytes: 10.16 MB; ratio: 7.51
    cparams := cparams(clevel=1, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999700e+06
     9.99999800e+06   9.99999900e+06]
  >>> bcolz.carray(a, bcolz.cparams(clevel=9))
  carray((10000000,), float64)
    nbytes: 76.29 MB; cbytes: 1.15 MB; ratio: 66.09
    cparams := cparams(clevel=9, shuffle=True, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999700e+06
     9.99999800e+06   9.99999900e+06]

Also, you can decide if you want to disable the shuffle filter that
comes with Blosc::

  >>> bcolz.carray(a, bcolz.cparams(shuffle=False))
  carray((10000000,), float64)
    nbytes: 76.29 MB; cbytes: 36.70 MB; ratio: 2.08
    cparams := cparams(clevel=5, shuffle=False, cname='blosclz')
  [  0.00000000e+00   1.00000000e+00   2.00000000e+00 ...,   9.99999700e+06
     9.99999800e+06   9.99999900e+06]

but, as can be seen, the compression ratio is much worse in this case.
In general it is recommend to let shuffle active (unless you are
fine-tuning the performance for an specific carray).

See :ref:`opt-tips` chapter for info on how you can change other
internal parameters like the size of the chunk.

Also, for setting globally or permanently your own defaults for the
compression parameters, see :ref:`defaults` chapter.


Accessing carray data
---------------------

The way to access carray data is very similar to the NumPy indexing
scheme, and in fact, supports all the indexing methods supported by
NumPy.

Specifying an index or slice::

  >>> a = np.arange(10)
  >>> b = bcolz.carray(a)
  >>> b[0]
  0
  >>> b[-1]
  9
  >>> b[2:4]
  array([2, 3])
  >>> b[::2]
  array([0, 2, 4, 6, 8])
  >>> b[3:9:3]
  array([3, 6])

Note that NumPy objects are returned as the result of an indexing
operation.  This is on purpose because normally NumPy objects are more
featured and flexible (specially if they are small).  In fact, a handy
way to get a NumPy array out of a carray object is asking for the
complete range::

  >>> b[:]
  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Fancy indexing is supported too.  For example, indexing with boolean
arrays gives::

  >>> barr = np.array([True]*5+[False]*5)
  >>> b[barr]
  array([0, 1, 2, 3, 4])
  >>> b[bcolz.carray(barr)]
  array([0, 1, 2, 3, 4])

Or, with a list of indices::

  >>> b[[2,3,0,2]]
  array([2, 3, 0, 2])
  >>> b[bcolz.carray([2,3,0,2])]
  array([2, 3, 0, 2])

Querying carrays
----------------

carrays can be queried in different ways.  The most easy (yet
powerful) way is by using its set of iterators::

  >>> a = np.arange(1e7)
  >>> b = bcolz.carray(a)
  >>> %time sum(v for v in a if v < 10)
  CPU times: user 1.82 s, sys: 47.8 ms, total: 1.87 s
  Wall time: 1.85 s
  45.0
  >>> %time sum(v for v in b if v < 10)
  CPU times: user 624 ms, sys: 12.3 ms, total: 637 ms
  Wall time: 605 ms # 3x faster than Numpy
  45.0

The iterator also has support for looking into slices of the array::

  >>> %time sum(v for v in b.iter(start=2, stop=20, step=3) if v < 10)
  CPU times: user 1.6 ms, sys: 560 µs, total: 2.16 ms
  Wall time: 1.35 ms
  15.0
  >>> %timeit sum(v for v in b.iter(start=2, stop=20, step=3) if v < 10)
  1000 loops, best of 3: 731 µs per loop

See that the time taken in this case is much shorter because the slice
to do the lookup is much shorter too.

Also, you can quickly retrieve the indices of a boolean carray that
have a true value::

  >>> barr = bcolz.eval("b<10")  # see 'Operating with carrays' section below
  >>> [i for i in barr.wheretrue()]
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  >>> %timeit [i for i in barr.wheretrue()]
  1000 loops, best of 3: 1.06 ms per loop

And get the values where a boolean array is true::

  >>> [i for i in b.where(barr)]
  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  >>> %timeit [i for i in b.where(barr)]
  100 loops, best of 3: 7.66 ms per loop

Note how `wheretrue` and `where` iterators are really fast.  They are
also very powerful.  For example, they support `limit` and `skip`
parameters for limiting the number of elements returned and skipping
the leading elements respectively::

  >>> [i for i in barr.wheretrue(limit=5)]
  [0, 1, 2, 3, 4]
  >>> [i for i in barr.wheretrue(skip=3)]
  [3, 4, 5, 6, 7, 8, 9]
  >>> [i for i in barr.wheretrue(limit=5, skip=3)]
  [3, 4, 5, 6, 7]

The advantage of the carray iterators is that you can use them in
generator contexts and hence, you don't need to waste memory for
creating temporaries, which can be important when dealing with large
arrays.

We have seen that this iterator toolset is very fast, so try to
express your problems in a way that you can use them extensively.

Modifying carrays
-----------------

Although it is a somewhat slow operation, carrays can be modified too.
You can do it by specifying scalar or slice indices::

  >>> a = np.arange(10)
  >>> b = bcolz.arange(10)
  >>> b[1] = 10
  >>> print b
  [ 0 10  2  3  4  5  6  7  8  9]
  >>> b[1:4] = 10
  >>> print b
  [ 0 10 10 10  4  5  6  7  8  9]
  >>> b[1::3] = 10
  >>> print b
  [ 0 10 10 10 10  5  6 10  8  9]

Modification by using fancy indexing is supported too::

  >>> barr = np.array([True]*5+[False]*5)
  >>> b[barr] = -5
  >>> print b
  [-5 -5 -5 -5 -5  5  6 10  8  9]
  >>> b[[1,2,4,1]] = -10
  >>> print b
  [ -5 -10 -10  -5 -10   5   6  10   8   9]

However, you must be aware that modifying a carray is expensive::

  >>> a = np.arange(1e7)
  >>> b = bcolz.carray(a)
  >>> %timeit a[2] = 3
  10000000 loops, best of 3: 94.4 ns per loop
  >>> %timeit b[2] = 3
  1000 loops, best of 3: 274 µs per loop # 2900x slower than NumPy

although modifying values in latest chunk is somewhat cheaper::

  >>> %timeit a[-1] = 3
  10000000 loops, best of 3: 95 ns per loop
  >>> %timeit b[-1] = 3
  100000 loops, best of 3: 9.66 µs per loop # 101x slower than NumPy

In general, you should avoid modifications (if you can) when using
carrays.

Multidimensional carrays
------------------------

You can create multidimensional carrays too.  Look at this example::

  >>> a = bcolz.zeros((2,3))
  carray((2, 3), float64)  nbytes: 48; cbytes: 3.98 KB; ratio: 0.01
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [[ 0.  0.  0.]
   [ 0.  0.  0.]]

So, you can access any element in any dimension::

  >>> a[1]
  array([ 0.,  0.,  0.])
  >>> a[1,::2]
  array([ 0., 0.])
  >>> a[1,1]
  0.0

As you see, multidimensional carrays support the same multidimensional
indexes than its NumPy counterparts.

Also, you can use the `reshape()` method to set your desired shape to
an existing carray::

  >>> b = bcolz.arange(12).reshape((3,4))
  >>> b
  carray((3,), ('int64',(4,)))  nbytes: 96; cbytes: 4.00 KB; ratio: 0.02
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

Iterators loop over the leading dimension::

  >>> [r for r in b]
  [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]

And you can select columns there by using another indirection level::

  >>> [r[2] for r in b]
  [2, 6, 10]

Above, the third column has been selected.  Although for this case the
indexing is easier::

  >>> b[:,2]
  array([ 2,  6, 10])

the iterator approach typically consumes less memory resources.

Operating with carrays
----------------------

Right now, you cannot operate with carrays directly (although that
might be implemented in the future)::

  >>> x = bcolz.arange(1e7)
  >>> x + x
  TypeError: unsupported operand type(s) for +:
  'carray.carrayExtension.carray' and 'carray.carrayExtension.carray'

Rather, you should use the `eval` function::

  >>> y = bcolz.eval("x + x")
  >>> y
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 2.64 MB; ratio: 28.88
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0.0, 2.0, 4.0, ..., 19999994.0, 19999996.0, 19999998.0]

You can also compute arbitrarily complex expressions in one shot::

  >>> y = bcolz.eval(".5*x**3 + 2.1*x**2")
  >>> y
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 38.00 MB; ratio: 2.01
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [0.0, 2.6, 12.4, ..., 4.9999976e+20, 4.9999991e+20, 5.0000006e+20]

Note how the output of `eval()` is also a carray object.  You can pass
other parameters of the carray constructor too.  Let's force maximum
compression for the output::

  >>> y = bcolz.eval(".5*x**3 + 2.1*x**2", cparams=bcolz.cparams(9))
  >>> y
  carray((10000000,), float64)  nbytes: 76.29 MB; cbytes: 35.66 MB; ratio: 2.14
    cparams := cparams(clevel=9, shuffle=True, cname='blosclz')
  [0.0, 2.6, 12.4, ..., 4.9999976e+20, 4.9999991e+20, 5.0000006e+20]

By default, `eval` will use Numexpr virtual machine if it is installed
and if not, it will default to use the Python one (via NumPy).  You
can use the `vm` parameter to select the desired virtual machine
("numexpr" or "python")::

  >>> %timeit bcolz.eval(".5*x**3 + 2.1*x**2", vm="numexpr")
  10 loops, best of 3: 303 ms per loop
  >>> %timeit bcolz.eval(".5*x**3 + 2.1*x**2", vm="python")
  10 loops, best of 3: 1.9 s per loop

As can be seen, using the "numexpr" virtual machine is generally
(much) faster, but there are situations that the "python" one is
desirable because it offers much more functionality::

  >>> bcolz.eval("diff(x)", vm="numexpr")
  NameError: variable name ``diff`` not found
  >>> bcolz.eval("np.diff(x)", vm="python")
  carray((9999389,), float64)  nbytes: 76.29 MB; cbytes: 814.25 KB; ratio: 95.94
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]

Finally, `eval` lets you select the type of the outcome to be a NumPy
array by using the `out_flavor` argument::

  >>> bcolz.eval("x**3", out_flavor="numpy")
  array([  0.00000000e+00,   1.00000000e+00,   8.00000000e+00, ...,
           9.99999100e+20,   9.99999400e+20,   9.99999700e+20])

For setting globally or permanently your own defaults for the `vm` and
`out_flavors`, see :ref:`defaults` chapter.

carray metadata
---------------

carray implements several attributes, like `dtype`, `shape` and `ndim`
that makes it to 'quack' like a NumPy array::

  >>> a = np.arange(1e7)
  >>> b = bcolz.carray(a)
  >>> b.dtype
  dtype('float64')
  >>> b.shape
  (10000000,)

In addition, it implements the `cbytes` attribute that tells how many
bytes in memory (or on-disk) uses the carray object::

  >>> b.cbytes
  2691722

This figure is approximate and it is generally lower than the original
(uncompressed) datasize can be accessed by using `nbytes` attribute::

  >>> b.nbytes
  80000000

which is the same than the equivalent NumPy array::

  >>> a.size*a.dtype.itemsize
  80000000

For knowing the compression level used and other optional filters, use
the `cparams` read-only attribute::

  >>> b.cparams
  cparams(clevel=5, shuffle=True, cname='blosclz')

Also, you can check which the default value is (remember, used when
`resize` -ing the carray)::

  >>> b.dflt
  0.0

You can access the `chunklen` (the length for each chunk) for this
carray::

  >>> b.chunklen
  16384

For a complete list of public attributes of carray, see section on
:ref:`carray-attributes`.

.. _carray-attrs:

carray user attrs
-----------------

Besides the regular attributes like `shape`, `dtype` or `chunklen`,
there is another set of attributes that can be added (and removed) by
the user in another name space.  This space is accessible via the
special `attrs` attribute, in the following example we will trigger flushing
data to disk manually::

  >>> a = bcolz.carray([1,2], rootdir='mydata')
  >>> a.attrs
  *no attrs*

As you see, by default there are no attributes attached to `attrs`.
Also, notice that the carray that we have created is persistent and
stored on the 'mydata' directory.  Let's add one attribute here::

  >>> a.attrs['myattr'] = 234
  >>> a.attrs
  myattr : 234

So, we have attached the 'myattr' attribute with the value 234.  Let's
add a couple of attributes more::

  >>> a.attrs['temp'] = 23 
  >>> a.attrs['unit'] = 'Celsius'
  >>> a.attrs
  unit : 'Celsius'
  myattr : 234
  temp : 23

good, we have three of them now.  You can attach as many as you want,
and the only current limitation is that they have to be serializable
via JSON.

As the 'a' carray is persistent, it can re-opened in other Python session::

  >>> a.flush()
  >>> ^D 
  $ python
  Python 2.7.3rc2 (default, Apr 22 2012, 22:30:17) 
  [GCC 4.6.3] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import carray as ca
  >>> a = bcolz.open(rootdir="mydata")
  >>> a                            # yeah, our data is back
  carray((2,), int64)
    nbytes: 16; cbytes: 4.00 KB; ratio: 0.00
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    rootdir := 'mydata'
  [1 2]
  >>> a.attrs                      # and so is user attrs!
  temp : 23
  myattr : 234
  unit : u'Celsius'

Now, let's remove a couple of user attrs::

  >>> del a.attrs['myattr']                           
  >>> del a.attrs['unit']
  >>> a.attrs
  temp : 23

So, it is really easy to make use of this feature so as to complement
your data with (potentially persistent) metadata of your choice.  Of
course, the `ctable` object also wears this capability.


Tutorial on ctable objects
==========================

The bcolz package comes with a handy object that arranges data by
column (and not by row, as in NumPy's structured arrays).  This allows
for much better performance for walking tabular data by column and
also for adding and deleting columns.

Creating a ctable
-----------------

You can build ctable objects in many different ways, but perhaps the
easiest one is using the `fromiter` constructor::

  >>> N = 100*1000
  >>> ct = bcolz.fromiter(((i,i*i) for i in xrange(N)), dtype="i4,f8", count=N)
  >>> ct
  ctable((100000,), |V12) nbytes: 1.14 MB; cbytes: 283.27 KB; ratio: 4.14
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [(0, 0.0), (1, 1.0), (2, 4.0), ...,
   (99997, 9999400009.0), (99998, 9999600004.0), (99999, 9999800001.0)]

You can also build an empty ctable with `bzolz.zeros` indicating zero length and
appending data afterwards, we encourage you to use the `with` statement for
this, it will take care of flushing data to disk once you are done appending
data.::

  >>> with bcolz.zeros(0, dtype="i4,f8", rootdir='mydir', mode="w") as ct:
  ...:     for i in xrange(N):
  ...:        ct.append((i, i**2))
  ...:
  >>> bcolz.ctable(rootdir='mydir') 
  ctable((100000,), [('f0', '<i4'), ('f1', '<f8')])
    nbytes: 1.14 MB; cbytes: 247.18 KB; ratio: 4.74
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
    rootdir := 'mydir'
  [(0, 0.0) (1, 1.0) (2, 4.0) ..., (99997, 9999400009.0)
   (99998, 9999600004.0) (99999, 9999800001.0)]

However, we can see how the latter approach does not compress as well.
Why?  Well, carray has machinery for computing 'optimal' chunksizes
depending on the number of entries.  For the first case, carray can
figure out the number of entries in final array, but not for the loop
case.  You can solve this by passing the final length with the
`expectedlen` argument to the ctable constructor::

  >>> ct = bcolz.zeros(0, dtype="i4,f8", expectedlen=N)
  >>> for i in xrange(N):
  ...:    ct.append((i, i**2))
  ...:
  >>> ct
  ctable((100000,), |V12) nbytes: 1.14 MB; cbytes: 283.27 KB; ratio: 4.14
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [(0, 0.0), (1, 1.0), (2, 4.0), ...,
   (99997, 9999400009.0), (99998, 9999600004.0), (99999, 9999800001.0)]

Okay, the compression ratio is the same now.

Accessing and setting rows
--------------------------

The ctable object supports the most common indexing operations in
NumPy::

  >>> ct[1]
  (1, 1.0)
  >>> type(ct[1])
  <type 'numpy.void'>
  >>> ct[1:6]
  array([(1, 1.0), (2, 4.0), (3, 9.0), (4, 16.0), (5, 25.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])

The first thing to have in mind is that, similarly to `carray`
objects, the result of an indexing operation is a native NumPy object
(in the case above a scalar and a structured array).

Fancy indexing is also supported::

  >>> ct[[1,6,13]]
  array([(1, 1.0), (6, 36.0), (13, 169.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])
  >>> ct["(f0>0) & (f1<10)"]
  array([(1, 1.0), (2, 4.0), (3, 9.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])

Note that conditions over columns are expressed as string expressions
(in order to use Numexpr under the hood), and that the column names
are understood correctly.

Setting rows is also supported::

  >>> ct[1] = (0,0)
  >>> ct
  ctable((100000,), |V12) nbytes: 1.14 MB; cbytes: 279.89 KB; ratio: 4.19
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [(0, 0.0), (0, 0.0), (2, 4.0), ...,
   (99997, 9999400009.0), (99998, 9999600004.0), (99999, 9999800001.0)]
  >>> ct[1:6]
  array([(0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])

And in combination with fancy indexing too::

  >>> ct[[1,6,13]] = (1,1)
  >>> ct[[1,6,13]]
  array([(1, 1.0), (1, 1.0), (1, 1.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])
  >>> ct["(f0>=0) & (f1<10)"] = (2,2)
  >>> ct[:7]
  array([(2, 2.0), (2, 2.0), (2, 2.0), (2, 2.0), (2, 2.0), (2, 2.0),
         (6, 36.0)],
        dtype=[('f0', '<i4'), ('f1', '<f8')])

As you may have noticed, fancy indexing in combination with conditions
is a very powerful feature.

Adding and deleting columns
---------------------------

Adding and deleting columns is easy and, due to the column-wise data
arrangement, very efficient.  Let's add a new column on an existing
ctable::

  >>> N = 100*1000
  >>> ct = bcolz.fromiter(((i,i*i) for i in xrange(N)), dtype="i4,f8", count=N)
  >>> new_col = np.linspace(0, 1, 100*1000)
  >>> ct.addcol(new_col)
  >>> ct
  ctable((100000,), |V20) nbytes: 1.91 MB; cbytes: 528.83 KB; ratio: 3.69
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [(0, 0.0, 0.0), (1, 1.0, 1.000010000100001e-05),
   (2, 4.0, 2.000020000200002e-05), ...,
   (99997, 9999400009.0, 0.99997999979999797),
   (99998, 9999600004.0, 0.99998999989999904), (99999, 9999800001.0, 1.0)]

Now, remove the already existing 'f1' column::

  >>> ct.delcol('f1')
  >>> ct
  ctable((100000,), |V12) nbytes: 1.14 MB; cbytes: 318.68 KB; ratio: 3.68
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [(0, 0.0), (1, 1.000010000100001e-05), (2, 2.000020000200002e-05), ...,
   (99997, 0.99997999979999797), (99998, 0.99998999989999904), (99999, 1.0)]

As said, adding and deleting columns is very cheap, so don't be afraid
of using them extensively.

Iterating over ctable data
--------------------------

You can make use of the `iter()` method in order to easily iterate
over the values of a ctable.  `iter()` has support for start, stop and
step parameters::

  >>> N = 100*1000
  >>> t = bcolz.fromiter(((i,i*i) for i in xrange(N)), dtype="i4,f8", count=N)
  >>> [row for row in ct.iter(1,10,3)]
  [row(f0=1, f1=1.0), row(f0=4, f1=16.0), row(f0=7, f1=49.0)]

Note how the data is returned as `namedtuple` objects of type
``row``.  This allows you to iterate the fields more easily by using
field names::

  >>> [(f0,f1) for f0,f1 in ct.iter(1,10,3)]
  [(1, 1.0), (4, 16.0), (7, 49.0)]

You can also use the ``[:]`` accessor to get rid of the ``row``
namedtuple, and return just bare tuples::

  >>> [row[:] for row in ct.iter(1,10,3)]
  [(1, 1.0), (4, 16.0), (7, 49.0)]

Also, you can select specific fields to be read via the `outcols`
parameter::

  >>> [row for row in ct.iter(1,10,3, outcols='f0')]
  [row(f0=1), row(f0=4), row(f0=7)]
  >>> [(nr,f0) for nr,f0 in ct.iter(1,10,3, outcols='nrow__,f0')]
  [(1, 1), (4, 4), (7, 7)]

Please note the use of the special 'nrow__' label for referring to
the current row.

Iterating over the output of conditions along columns
-----------------------------------------------------

One of the most powerful capabilities of the ctable is the ability to
iterate over the rows whose fields fulfill some conditions (without
the need to put the results in a NumPy container, as described in the
"Accessing and setting rows" section above).  This can be very useful
for performing operations on very large ctables without consuming lots
of storage space.

Here it is an example of use::

  >>> N = 100*1000
  >>> t = bcolz.fromiter(((i,i*i) for i in xrange(N)), dtype="i4,f8", count=N)
  >>> [row for row in ct.where("(f0>0) & (f1<10)")]
  [row(f0=1, f1=1.0), row(f0=2, f1=4.0), row(f0=3, f1=9.0)]
  >>> sum([row.f1 for row in ct.where("(f1>10)")])
  3.3333283333312755e+17

And by using the `outcols` parameter, you can specify the fields that
you want to be returned::

  >>> [row for row in ct.where("(f0>0) & (f1<10)", "f1")]
  [row(f1=1.0), row(f1=4.0), row(f1=9.0)]


You can even specify the row number fulfilling the condition::

  >>> [(f1,nr) for f1,nr in ct.where("(f0>0) & (f1<10)", "f1,nrow__")]
  [(1.0, 1), (4.0, 2), (9.0, 3)]

Performing operations on ctable columns
---------------------------------------

The ctable object also wears an `eval()` method that is handy for
carrying out operations among columns::

  >>> ct.eval("cos((3+f0)/sqrt(2*f1))")
  carray((1000000,), float64)  nbytes: 7.63 MB; cbytes: 2.23 MB; ratio: 3.42
    cparams := cparams(clevel=5, shuffle=True, cname='blosclz')
  [nan, -0.951363128126, -0.195699435691, ...,
   0.760243218982, 0.760243218983, 0.760243218984]

Here, one can see an exception in ctable methods behaviour: the
resulting output is a ctable, and not a NumPy structured array.  This
is so because the output of `eval()` is of the same length than the
ctable, and thus it can be pretty large, so compression maybe of help
to reduce its storage needs.


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

  - `Bdot`: provides big dot products (by making your RAM bigger on
      the inside).  Supports ``matrix . vector`` and ``matrix
      . matrix`` for most common numpy numeric data types. See
      https://github.com/tailwind/bdot

Though not a extensions itself, it is worth pointing out `Dask`. Dask
plays nicely with bcolz and provides multi-core execution on
larger-than-memory datasets using blocked algorithms and task
scheduling. See https://github.com/ContinuumIO/dask.

In addition, bcolz also interacts well with `itertools`, `Pytoolz` or
`Cytoolz` too and they might offer you already the amount of
performance and functionality you are after.

In the next section we will go through all the steps needed to write
your own extension on top of bcolz.

How to use bcolz as part of the infrastructure
----------------------------------------------

Go to the root directory of bcolz, inside ``doc/my_package/`` you will
find a small extension example.

Before you can run this example you will need to install the following
packages.  Run ``pip install cython``, ``pip install numpy`` and ``pip
install bcolz`` to install these packages.  In case you prefer Conda
package management system execute ``conda install cython numpy bcolz``
and you should be ready to go.  See ``requirements.txt``:

.. literalinclude:: my_package/requirements.txt
    :language: python

Once you have those packages installed, change your working directory
to ``doc/my_package/``, please see `pkg. example
<https://github.com/Blosc/bcolz/tree/master/doc/my_package>`_ and run
``python setup.py build_ext --inplace`` from the terminal, if
everything ran smoothly you should be able to see a binary file
``my_extension/example_ext.so`` next to the ``.pyx`` file.

If you have any problems compiling these extensions, please make sure
your bcolz version is at least ``0.8.0``, previous versions don't
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
