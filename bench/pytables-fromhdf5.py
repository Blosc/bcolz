# Benchmark for evaluate best ways to read from a PyTables Table

import bcolz
import tables as tb
import numpy as np
from time import time

filepath = 'fromhdf5.h5'
nodepath = '/ctable'
NR = int(1e6)
NC = 10
dsize = (NR * NC * 4) / 2. ** 30

bcolz.cparams.setdefaults(clevel=5)

a = bcolz.arange(NR, dtype='i4')
#ra = np.rec.fromarrays([a]*NC, names=['f%d'%i for i in range(NC)])
ra = bcolz.ctable((a,)*NC)[:]

t0 = time()
f = tb.open_file(filepath, "w")
f.create_table(f.root, nodepath[1:], ra)
f.close()
tt = time() - t0
print("time for storing the HDF5 table: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using an iterator
t0 = time()
f = tb.open_file(filepath)
t = f.get_node(nodepath)
t = bcolz.fromiter((r[:] for r in t), dtype=t.dtype, count=len(t))
f.close()
tt = time() - t0
print("time with fromiter: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using blocked read
t0 = time()
f = tb.open_file(filepath)
t = f.get_node(nodepath)
names = t.colnames
dtypes = [dt[0] for dt in t.dtype.fields.values()]
cols = [np.zeros(0, dtype=dt) for dt in dtypes]
ct = bcolz.ctable(cols, names)
bs = t._v_chunkshape[0]
for i in xrange(0, len(t), bs):
    ct.append(t[i:i+bs])
f.close()
tt = time() - t0
print("time with blocked read: %.2f (%.2f GB/s)" % (tt, dsize / tt))

# Using generic implementation
t0 = time()
t = bcolz.ctable.fromhdf5(filepath, nodepath)
tt = time() - t0
print("time with fromhdf5: %.2f (%.2f GB/s)" % (tt, dsize / tt))


#print(repr(ct))

