import bcolz
import numpy
from .bench_helper import ctime

N = 1000 * 1000

ct = bcolz.fromiter(((i, i * i, i * i * i)
                     for i in xrange(N)), dtype='i8,i8,i8', count=N)

b = numpy.array(numpy.arange(N) % 2, dtype="bool")
c = bcolz.carray(b)

sorted_index = range(1, N, 2)


class Suite:

    def time_tolist(self):
        return (ct['f0'][sorted_index]).tolist()

    def time_where_01(self):
        return [x.f0 for x in ct.where(b)]

    def time_where_02(self):
        return [x.f0 for x in ct.where(c)]

    def time_where_03(self):
        return [x for x in ct['f0'].where(b)]

    def time_where_04(self):
        return [x for x in ct['f0'].where(c)]

    def time_sum_01(self):
        return sum([x for x in ct['f0'].where(c)])

    def time_sum_02(self):
        return sum(x for x in ct['f0'].where(c))

    def time_sum_03(self):
        return bcolz.fromiter((x for x in ct['f0'].where(c)),
                              dtype=ct['f0'].dtype, count=c.wheretrue().sum()).sum()

    def time_sum_na_01(self):
        # sum with no NA's
        # x==x check to leave out NA values
        return sum([x for x in ct['f0'].where(c) if x == x])

    def time_sum_na_02(self):
        # sum with no NA's
        # x==x check to leave out NA values
        return sum((x for x in ct['f0'].where(c) if x == x))

if __name__ == '__main__':
    s = Suite()
    with ctime("time_sum_01"):
        s.time_sum_01()
    with ctime("time_sum_02"):
        s.time_sum_02()
    with ctime("time_sum_03"):
        s.time_sum_03()
    with ctime("time_sum_na_01"):
        s.time_sum_na_01()
    with ctime("time_sum_na_02"):
        s.time_sum_na_02()
    with ctime("time_tolist"):
        s.time_tolist()
    with ctime("time_where_01"):
        s.time_where_01()
    with ctime("time_where_02"):
        s.time_where_02()
    with ctime("time_where_03"):
        s.time_where_03()
    with ctime("time_where_04"):
        s.time_where_04()
