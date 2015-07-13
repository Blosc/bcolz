# -*- coding: utf-8 -*-
import numpy as np

from bcolz.tests.common import TestCase
from bcolz.utils import to_ndarray


class tondarrayTest(TestCase):

    def test_dtype_None(self):
        array = np.array([[0, 1, 2], [2, 1, 0]]).T
        self.assertTrue(to_ndarray(array, None, safe=True).flags.contiguous,
                        msg='to_ndarray: Non contiguous arrays are not being consolidated when dtype is None')
