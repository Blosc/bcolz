from __future__ import print_function
import contextlib, time
import bcolz, numpy

@contextlib.contextmanager
def ctime(label=""):
    "Counts the time spent in some context"
    t = time.time()
    yield
    print(label, round(time.time() - t, 3), "sec")


N = 1000 * 1000

ct = bcolz.fromiter(((i, i*i, i*i*i) for i in xrange(N)), dtype='i8,i8,i8', count=N)

b = numpy.array(numpy.arange(N) % 2, dtype="bool")
c = bcolz.carray(b)

sorted_index = range(1, N, 2)
with ctime():
    r0 = (ct['f0'][sorted_index]).tolist()

with ctime():
    r1 = [x.f0 for x in ct.where(b)]
assert r0 == r1

with ctime():
    r2 = [x.f0 for x in ct.where(c)]
assert r0 == r2

with ctime():
    r3 = [x for x in ct['f0'].where(b)]
assert r0 == r3

with ctime():
    r4 = [x for x in ct['f0'].where(c)]
assert r0 == r4

# sum
with ctime("sum list"):
    r5 = sum([x for x in ct['f0'].where(c)])

with ctime("sum generator"):
    r6 = sum(x for x in ct['f0'].where(c))
assert r5 == r6

with ctime("sum method"):
    r7 = bcolz.fromiter((x for x in ct['f0'].where(c)),
                        dtype=ct['f0'].dtype,
                        count=c.wheretrue().sum()).sum()
assert r7 == r5

# sum with no NA's
with ctime("sum with no NA (list)"):
    r8 = sum([x for x in ct['f0'].where(c) if x == x])  # x==x check to leave out NA values

# sum with no NA's
with ctime("sum with no NA (generator)"):
    r9 = sum((x for x in ct['f0'].where(c) if x == x))  # x==x check to leave out NA values
