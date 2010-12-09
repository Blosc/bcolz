# Benchmark to compare the times for querying ctable objects.  Numexpr
# is needed in order to execute this.  A comparison with PyTables (if
# installed) is also done.

import sys, math
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import carray as ca
from time import time

NR = 1e6 # the number of rows
NC = 100      # the number of columns
clevel = 5    # the compression level
#squery = "((((f0+f1)<3) & ((f30+f31)<3e3)) | ((f2>3) & (f2<3e4)))"  # the query
#nquery = "((((t['f0']+t['f1'])<3) & ((t['f30']+t['f31'])<3e3)) | ((t['f2']>3) & (t['f2']<3e4)))"  # the query for a recarray
squery = "(f1>300) & ((f2>3) & (f2<1e4))"  # the query
nquery = "(t['f1']>300) & ((t['f2']>3) & (t['f2']<1e4))"  # for a recarray

print "Creating inputs..."

x = np.arange(NR)
#x = np.linspace(0,100,NR)
# Create a recarray made by copies of x
t = np.rec.fromarrays([x]*NC)

print "Querying '%s' with 10^%d rows" % (squery, int(math.log10(NR)))

t0 = time()
out = t[eval(nquery)][['f0','f2']]
print "Time for plain numpy --> %.3f" % (time()-t0,)

map_field = dict(("f%s"%i, t["f%s"%i]) for i in range(NC))
t0 = time()
out = t[ne.evaluate(squery, map_field)][['f0','f2']]
print "Time for numexpr --> %.3f" % (time()-t0,)

# Seems that this works better if we dividw the number of cores by 2.
# Maybe due to some contention between Numexpr and Blosc?
ca.set_nthreads(ca.ncores//2)

# Create a ctable out of the recarray
tc = ca.ctable(t,  cparams=ca.cparams(0))

t0 = time()
#cout = tc.eval(sexpr, cparams=cparams)
cout = [row for row in tc.where(squery, ['f0','f2'])]
print "Time for ctable (uncompressed) --> %.3f" % (time()-t0,),
print "-- size (MB):", tc.cbytes / 2**20

# Create a ctable out of the recarray
tc = ca.ctable(t, cparams=ca.cparams(clevel))

t0 = time()
#cout = tc.eval(sexpr, cparams=cparams)
cout = [row for row in tc.where(squery, ['f0','f2'])]
print "Time for ctable (compressed) --> %.3f" % (time()-t0,),
print " -- size (MB):", tc.cbytes / 2**20

try:
    import tables
except:
    sys.exit()

f = tables.openFile("pytables.h5", "w")
tpt = f.createTable(f.root, 'tpt', t, filters=tables.Filters(clevel, 'blosc'))

t0 = time()
cout = tpt.readWhere(squery)[['f0','f2']]
print "Time for PyTables (non-indexed) --> %.3f" % (time()-t0,)

if not tables.__version__.endswith("pro"):
    f.close()
    sys.exit()

print "PyTables Pro detected!  Indexing f2 column..."
# Index the column for maximum speed
tpt.cols.f2.createCSIndex()

t0 = time()
cout = tpt.readWhere(squery)[['f0','f2']]
print "Time for PyTables Pro (indexed) --> %.3f" % (time()-t0,)

f.close()

