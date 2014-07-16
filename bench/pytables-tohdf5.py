# Benchmark for evaluate best ways to write to a PyTables Table

import os
import bcolz
import tables as tb
import numpy as np
from time import time

filepath = 'tohdf5.h5'
nodepath = '/ctable'
NR = int(1e6)
NC = 10
dsize = (NR * NC * 4) / 2. ** 30

bcolz.cparams.setdefaults(clevel=5)

a = bcolz.arange(NR, dtype='i4')
ct = bcolz.ctable((a,)*NC)

# Row-by-row using an iterator
# t0 = time()
# f = tb.open_file(filepath, 'w')
# t = f.create_table(f.root, nodepath[1:], ct.dtype)
# for row in ct:
#     t.append([row])
# f.close()
# tt = time() - t0
# print("time with iterator: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using blocked write
t0 = time()
f = tb.open_file(filepath, 'w')
t = f.create_table(f.root, nodepath[1:], ct.dtype)
for block in bcolz.iterblocks(ct):
    t.append(block)
f.close()
tt = time() - t0
print("time with blocked write: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using generic implementation
os.remove(filepath)
t0 = time()
#ct.tohdf5(filepath, nodepath)
ct.tohdf5(filepath, nodepath, cname="blosc:blosclz")
tt = time() - t0
print("time with tohdf5: %.2f (%.2f GB/s)" % (tt, dsize / tt))


#print(repr(ct))

