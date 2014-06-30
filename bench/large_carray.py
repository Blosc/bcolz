# # Benchmark to check the creation of an array of length > 2**32 (5e9)

import sys
from time import time

import bcolz

if sys.version_info >= (3,0):
    long = int


t0 = time()
#cn = bcolz.zeros(5e9, dtype="i1")
cn = bcolz.zeros(5e9, dtype="i1", rootdir='large_carray-bench', mode='w')
print("Creation time:", round(time() - t0, 3))
print("len:", len(cn))
assert len(cn) == int(5e9)

t0 = time()
cn = bcolz.carray(rootdir='large_carray-bench', mode='a')
print("Re-open time:", round(time() - t0, 3))
print("len(cn)", len(cn))
assert len(cn) == int(5e9)

# Now check some accesses
cn[1] = 1
assert cn[1] == 1
cn[int(2e9)] = 2
assert cn[int(2e9)] == 2
cn[long(3e9)] = 3
assert cn[long(3e9)] == 3
cn[-1] = 4
assert cn[-1] == 4

t0 = time()
assert cn.sum() == 10
print("Sum time:", round(time() - t0, 3))

print("str(carray):", str(cn))
print("repr(carray):", repr(cn))
