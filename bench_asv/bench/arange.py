import numpy as np
import bcolz
from .bench_helper import ctime


class Suite:

    def setup(self):
        self.N = 1e8
        self.dtype = 'i4'
        self.start = 5
        self.stop = self.N
        self.step = 4
        self.a = np.arange(self.start, self.stop, self.step, dtype=self.dtype)

    def time_arange(self):
        ac = bcolz.arange(self.start, self.stop, self.step, dtype=self.dtype)

if __name__ == '__main__':
    suite = Suite()
    suite.setup()
    with ctime("time_arange"):
        suite.time_arange()
