# -*- coding: utf-8 -*-
########################################################################
#
#       License: BSD
#       Created: August 17, 2012
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

from __future__ import absolute_import

import os
import tempfile

import bcolz
from bcolz.tests.common import (
    MayBeDiskTest, TestCase, unittest, skipUnless)


class basicTest(MayBeDiskTest):

    def getobject(self):
        if self.flavor == 'carray':
            obj = bcolz.zeros(10, dtype="i1", rootdir=self.rootdir)
            assert type(obj) == bcolz.carray
        elif self.flavor == 'ctable':
            obj = bcolz.fromiter(((i, i*2) for i in range(10)), dtype='i2,f4',
                                 count=10, rootdir=self.rootdir)
            assert type(obj) == bcolz.ctable
        return obj

    def test00a(self):
        """Creating attributes in a new object."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        self.assertTrue(cn.attrs['attr1'] == 'val1')
        self.assertTrue(cn.attrs['attr2'] == 'val2')
        self.assertTrue(cn.attrs['attr3'] == 'val3')
        self.assertTrue(len(cn.attrs) == 3)

    def test00b(self):
        """Accessing attributes in a opened object."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Re-open the object
        if self.rootdir:
            cn = bcolz.open(rootdir=self.rootdir)
        self.assertTrue(cn.attrs['attr1'] == 'val1')
        self.assertTrue(cn.attrs['attr2'] == 'val2')
        self.assertTrue(cn.attrs['attr3'] == 'val3')
        self.assertTrue(len(cn.attrs) == 3)

    def test01a(self):
        """Removing attributes in a new object."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Remove one of them
        del cn.attrs['attr2']
        self.assertTrue(cn.attrs['attr1'] == 'val1')
        self.assertTrue(cn.attrs['attr3'] == 'val3')
        self.assertRaises(KeyError, cn.attrs.__getitem__, 'attr2')
        self.assertTrue(len(cn.attrs) == 2)

    def test01b(self):
        """Removing attributes in a opened object."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        # Reopen
        if self.rootdir:
            cn = bcolz.open(rootdir=self.rootdir)
        # Remove one of them
        del cn.attrs['attr2']
        self.assertTrue(cn.attrs['attr1'] == 'val1')
        self.assertTrue(cn.attrs['attr3'] == 'val3')
        self.assertRaises(KeyError, cn.attrs.__getitem__, 'attr2')
        self.assertTrue(len(cn.attrs) == 2)

    def test01c(self):
        """Appending attributes in a opened object."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        # Reopen
        if self.rootdir:
            cn = bcolz.open(rootdir=self.rootdir)
        # Append attrs
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        self.assertTrue(cn.attrs['attr1'] == 'val1')
        self.assertTrue(cn.attrs['attr2'] == 'val2')
        self.assertTrue(cn.attrs['attr3'] == 'val3')
        self.assertTrue(len(cn.attrs) == 3)

    def test02(self):
        """Checking iterator in attrs accessor."""

        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 2
        cn.attrs['attr3'] = 3.
        count = 0
        for key, val in cn.attrs:
            if key == 'attr1':
                self.assertEqual(val, 'val1')
            if key == 'attr2':
                self.assertEqual(val, 2)
            if key == 'attr3':
                self.assertEqual(val, 3.)
            count += 1
        self.assertTrue(count, 3)

    @skipUnless(bcolz.tables_here, "PyTables not here")
    def test03(self):
        """Checking roundtrip of attrs in HDF5 files."""

        if self.flavor != 'ctable':
            # ATM HDF5 serialization only works for ctables object
            return
        cn = self.getobject()
        # Some attrs
        cn.attrs['attr1'] = 'val1'
        cn.attrs['attr2'] = 'val2'
        cn.attrs['attr3'] = 'val3'
        tmpfile = tempfile.mktemp(".h5")
        cn.tohdf5(tmpfile)
        cn = bcolz.ctable.fromhdf5(tmpfile)
        os.remove(tmpfile)
        self.assertEqual(cn.attrs['attr1'], 'val1')
        self.assertEqual(cn.attrs['attr2'], 'val2')
        self.assertEqual(cn.attrs['attr3'], 'val3')
        self.assertEqual(len(cn.attrs), 3)


class carrayTest(basicTest, TestCase):
    flavor = "carray"
    disk = False


class carrayDiskTest(basicTest, TestCase):
    flavor = "carray"
    disk = True


class ctableTest(basicTest, TestCase):
    flavor = "ctable"
    disk = False


class ctableDiskTest(basicTest, TestCase):
    flavor = "ctable"
    disk = True


if __name__ == '__main__':
    unittest.main(verbosity=2)


# Local Variables:
# mode: python
# tab-width: 4
# fill-column: 72
# End:
