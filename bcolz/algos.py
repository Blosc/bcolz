from collections import OrderedDict
from bcolz import carray


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
