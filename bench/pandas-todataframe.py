# Benchmark for evaluate best ways to convert into a pandas dataframe

import bcolz
import pandas as pd
from time import time

NR = int(1e6)
NC = 100

bcolz.cparams.setdefaults(clevel=0)
a = bcolz.arange(NR, dtype='i4')
t = bcolz.ctable((a,)*NC)

dsize = (NR * NC * 4) / 2. ** 30

# Adding a column once a time
t0 = time()
tnames = list(t.names)
firstk = tnames.pop(0)
df = pd.DataFrame.from_items([(firstk, t[firstk][:])])
for key in tnames:
    df[key] = t[key][:]
tt = time() - t0
print("time with from_items (adding cols): %.2f (%.2f GB/s)" % (tt, dsize / tt))
del df

# Using a generator
t0 = time()
df = pd.DataFrame.from_items(((key, t[key][:]) for key in t.names))
tt = time() - t0
print("time with from_items: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using generic implementation
t0 = time()
df = t.todataframe()
tt = time() - t0
print("time with todataframe: %.2f (%.2f GB/s)" % (tt, dsize / tt))


#print(df)

