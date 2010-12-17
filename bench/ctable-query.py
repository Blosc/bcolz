# Benchmark to compare the times for querying ctable objects.  Numexpr
# is needed in order to execute this.  A comparison with SQLite3 and
# PyTables (if installed) is also done.

import sys, math
import os, os.path
import subprocess
import getopt

import sqlite3
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numexpr as ne
import carray as ca
from time import time

NR = 1e6      # the number of rows
NC = 100      # the number of columns
mv = 1e10     # the mean value for entries (sig digits = 17 - log10(mv))
clevel = 3    # the compression level
show = False  # show statistics
# The query for a ctable
squery = "(f1>.9) & ((f2>.3) & (f2<.4))"  # the ctable query
# The query for a recarray
nquery = "(t['f1']>.9) & ((t['f2']>.3) & (t['f2']<.4))"  # for a recarray

tref = 0

def show_stats(explain, tref):
    "Show the used memory (only works for Linux 2.6.x)."
    # Build the command to obtain memory info
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1]) // 1024
        elif line.startswith("VmRSS:"):
            vmrss = int(line.split()[1]) // 1024
        elif line.startswith("VmData:"):
            vmdata = int(line.split()[1]) // 1024
        elif line.startswith("VmStk:"):
            vmstk = int(line.split()[1])
        elif line.startswith("VmExe:"):
            vmexe = int(line.split()[1])
        elif line.startswith("VmLib:"):
            vmlib = int(line.split()[1]) // 1024
    sout.close()
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s MB\tVmRSS: %7s MB" % (vmsize, vmrss)
    print "VmData: %7s MB\tVmStk: %7s KB" % (vmdata, vmstk)
    print "VmExe:  %7s KB\tVmLib: %7s MB" % (vmexe, vmlib)
    tnow = time()
    #print "WallClock time:", round(tnow - tref, 3)
    return tnow


def enter():
    global tref
    if show:
        tref = show_stats("Before creation", time())

def after_create():
    global tref
    if show:
        tref = show_stats("After creation", tref)

def after_query():
    global tref
    if show:
        show_stats("After query", tref)


def test_numpy():
    enter()
    t0 = time()
    np.random.seed(12)  # so as to get reproducible results
    t = np.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                    dtype="f8,"*NC)
    print "Time (creation) --> %.3f" % (time()-t0,)
    after_create()

    t0 = time()
    # out = t[eval(nquery)][['f0','f2']]
    out = np.fromiter(((row['f0'],row['f1']) for row in t[eval(nquery)]),
                      dtype="f8,f8")
    print "Time (query) --> %.3f" % (time()-t0,)
    after_query()
    return out


def test_numexpr():
    enter()
    t0 = time()
    np.random.seed(12)  # so as to get reproducible results
    t = np.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                    dtype="f8,"*NC)
    print "Time (creation) --> %.3f" % (time()-t0,)
    after_create()

    map_field = dict(("f%s"%i, t["f%s"%i]) for i in range(NC))
    t0 = time()
    #out = t[ne.evaluate(squery, map_field)][['f0','f2']]
    out = np.fromiter(((row['f0'],row['f1']) for row in
                       t[ne.evaluate(squery, map_field)]),
                      dtype="f8,f8")
    print "Time (query) --> %.3f" % (time()-t0,)
    after_query()
    return out


def test_ctable(clevel):
    enter()
    t0 = time()
    tc = ca.fromiter((mv+np.random.rand(NC)-mv for i in xrange(int(NR))),
                     dtype="f8,"*NC,
                     cparams=ca.cparams(clevel),
                     expectedlen=NR)
    print "Time (creation, clevel=%d) --> %.3f" % (clevel, time()-t0,)
    after_create()

    t0 = time()
    out = np.fromiter((row for row in tc.where(squery, ['f0','f2'])),
                      dtype="f8,f8")
    print "Time for (query, clevel=%d) --> %.3f" % (clevel, time()-t0,),
    print "-- size (MB):", tc.cbytes / 2**20
    after_query()
    return out


def test_sqlite(memory=True):
    enter()
    sqlquery = "(f1>.9) & ((f2>.3) & (f2<.4))"  # the query

    if memory:
        con = sqlite3.connect(":memory:")
    else:
        filename = "bench.sqlite"
        if os.path.exists(filename):
            os.remove(filename)
        con = sqlite3.connect(filename)

    # Create table
    t0 = time()
    fields = "(%s)" % ",".join(["f%d real"%i for i in range(NC)])
    con.execute("create table bench %s" % fields)

    # Insert a NR rows of data
    vals = "(%s)" % ",".join(["?" for i in range(NC)])
    with con:
        con.executemany("insert into bench values %s" % vals,
                        (mv+np.random.rand(NC)-mv for i in xrange(int(NR))))
    print "Time (creation) --> %.3f" % (time()-t0,)
    after_create()

    t0 = time()
    out = np.fromiter(
        (row for row in con.execute(
        "select f0, f2 from bench where %s" % sqlquery)),
        dtype="f8,f8")
    print "Time (query, non-indexed) --> %.3f" % (time()-t0,)
    after_query()

    # Create indexes
    t0 = time()
    con.execute("create index f1idx on bench (f1)")
    con.execute("create index f2idx on bench (f2)")
    print "Time (indexing) --> %.3f" % (time()-t0,)
    after_create()

    t0 = time()
    out = np.fromiter(
        (row for row in con.execute(
        "select f0, f2 from bench where %s" % sqlquery)),
        dtype="f8,f8")
    print "Time (query, indexed) --> %.3f" % (time()-t0,)
    after_query()

    return out


def test_pytables(clevel):
    enter()
    np.random.seed(12)  # so as to get reproducible results
    try:
        import tables
    except:
        sys.exit()

    f = tables.openFile("pytables.h5", "w")

    t0 = time()
    t = f.createTable(f.root, 'tpt', np.dtype("f8,"*NC),
                      filters=tables.Filters(clevel, 'blosc'),
                      expectedrows=NR)
    row = t.row
    for i in xrange(int(NR)//1000):
        t.append(mv+np.random.rand(1000,NC)-mv)
        t.flush()
    print "Time for PyTables (creation) --> %.3f" % (time()-t0,)
    after_create()

    t0 = time()
    out = np.fromiter(
        ((row['f1'],row['f2']) for row in t.where(squery)),
        dtype="f8,f8")
    print "Time for PyTables (query, non-indexed) --> %.3f" % (time()-t0,)
    after_query()

    if not tables.__version__.endswith("pro"):
        f.close()
        return

    print "PyTables Pro detected.  Indexing columns..."
    # Index the column for maximum speed
    t0 = time()
    t.cols.f1.createCSIndex()
    t.cols.f2.createCSIndex()
    print "Time (indexing) --> %.3f" % (time()-t0,)
    after_create()

    t0 = time()
    out = np.fromiter(
        ((row['f1'],row['f2']) for row in t.where(squery)),
        dtype="f8,f8")
    print "Time for PyTables Pro (query, indexed) --> %.3f" % (time()-t0,)
    after_query()

    f.close()
    return out

if __name__=="__main__":

    usage = """usage: %s [-s] [-m method] [-c ncols] [-r nrows] [-z clevel]
            -s show memory statistics (only for Linux)
            -m select the method: "ctable" (def.), "numpy", "numexpr", "sqlite"
            -c the number of columns in table (def. 100)
            -r the number of rows in table (def. 1e6)
            -z the compression level (def. 3)
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'sc:r:m:z:')
    except:
        sys.stderr.write(usage)
        sys.exit(1)

    method = "ctable"
    # Get the options
    for option in opts:
        if option[0] == '-s':
            show = True
        elif option[0] == '-m':
            method = option[1]
        elif option[0] == '-c':
            NC = int(option[1])
        elif option[0] == '-r':
            NR = float(option[1])
        elif option[0] == '-z':
            clevel = int(option[1])

    np.random.seed(12)  # so as to get reproducible results

    print "########## Checking method: %s ############" % method

    print "Querying '%s' with 10^%d rows and %d cols" % \
          (squery, int(math.log10(NR)), NC)
    print "Building database.  Wait please..."

    if method == "ctable":
        test_ctable(clevel)
    elif method == "numpy":
        test_numpy()
    elif method == "numexpr":
        test_numexpr()
    elif method == "sqlite":
        test_sqlite()
    elif method == "pytables":
        test_pytables(clevel)
