from collections import OrderedDict
from bcolz import carray
import numpy as np


def factorize_pure(carray_):
    count = 0
    lookup = OrderedDict()
    n = len(carray_)
    labels = carray([], dtype='uint64', expectedlen=n)

    for element in carray_[:]:
        try:
            idx = lookup[element]
        except KeyError:
            lookup[element] = idx = count
            count += 1
        labels.append(idx)

    return labels, lookup


def factorize_pure2(carray_):
    count = 0
    lookup = OrderedDict()
    n = len(carray_)
    labels = carray([], dtype='uint64', expectedlen=n)

    buffer = np.empty(len(carray_.chunks[0][:]), dtype='uint64')

    for chunk in carray_.chunks:

        for i, element in enumerate(chunk[:]):
            try:
                idx = lookup[element]
            except KeyError:
                lookup[element] = idx = count
                count += 1
            buffer[i] = idx
        labels.append(buffer)

    return labels, lookup
