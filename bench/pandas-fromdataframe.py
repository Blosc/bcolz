# Benchmark for evaluate best ways to convert from a pandas dataframe

from collections import OrderedDict
import bcolz
import pandas as pd
import numpy as np
from time import time

NR = int(1e6)
NC = 100

#bcolz.cparams.setdefaults(clevel=0)

print("Creating inputs...")
a = bcolz.arange(NR, dtype='i4')
df = pd.DataFrame.from_dict(OrderedDict(('f%d'%i, a[:]) for i in range(NC)))

dsize = (NR * NC * 4) / 2. ** 30

# Adding a column once a time
t0 = time()
names = list(df.columns.values)
firstk = names.pop(0)
t = bcolz.ctable([df[firstk]], names=(firstk,))
for key in names:
    t.addcol(np.array(df[key]), key)
tt = time() - t0
print("time with adding cols: %.2f (%.2f GB/s)" % (tt, dsize / tt))
del t

# Using an iterator
t0 = time()
names = list(df.columns.values)
t = bcolz.ctable([df[key] for key in names], names)
tt = time() - t0
print("time with constructor: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using generic implementation
t0 = time()
t = bcolz.ctable.fromdataframe(df)
tt = time() - t0
print("time with fromdataframe: %.2f (%.2f GB/s)" % (tt, dsize / tt))


#print(repr(t))

