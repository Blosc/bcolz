# Benchmark for evaluate best ways to convert from a pandas dataframe
# (version with a mix of columns of ints and strings)

import bcolz
import pandas as pd
import numpy as np
from time import time

NR = int(1e6)
NC = 100

#bcolz.cparams.setdefaults(clevel=0)

print("Creating inputs...")
a = bcolz.arange(NR, dtype='i4')
s = bcolz.fromiter(("%d"%i for i in xrange(NR)), dtype='S7', count=NR)
df = pd.DataFrame.from_items((
    ('f%d'%i, a[:] if i < (NC//2) else s[:]) for i in range(NC)))

dsize = (NR * (NC//2) * (a.dtype.itemsize + s.dtype.itemsize)) / 2. ** 20

print("Performing benchmarks...")
# # Using an iterator (will get objects)
# t0 = time()
# names = list(df.columns.values)
# t = bcolz.ctable([df[key] for key in names], names)
# tt = time() - t0
# print("time with constructor: %.2f (%.2f MB/s)" % (tt, dsize / tt))
# print(repr(t.dtype))

# Using generic implementation
t0 = time()
t = bcolz.ctable.fromdataframe(df)
tt = time() - t0
print("time with fromdataframe: %.2f (%.2f MB/s)" % (tt, dsize / tt))
print(t.dtype)

