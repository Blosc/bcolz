# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_array_equal

from bcolz.tests.common import TestCase
from bcolz.utils import to_ndarray


class tondarrayTest(TestCase):

    def test_dtype_None(self):
        array = np.array([[0, 1, 2], [2, 1, 0]]).T
        assert_array_equal(array, to_ndarray(array, None, safe=True),
                           'to_ndarray: Non contiguous arrays are not being consolidated when dtype is None')
