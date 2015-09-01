########################################################################
#
#       License: BSD
#       Created: July 15, 2014
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""Defaults for different bcolz parameters.
"""

from __future__ import absolute_import

import bcolz


class Defaults(object):
    """Class to taylor the setters and getters of default values."""

    def __init__(self):
        self.choices = {}

        # Choices setup
        self.choices['eval_out_flavor'] = ("carray", "numpy")
        self.choices['eval_vm'] = ("numexpr", "python")

    def check_choices(self, name, value):
        if value not in self.choices[name]:
            raise ValueError(
                "'%s' is incorrect value for '%s' default" % (value, name))

    def check_cparams(self, value):
        if not isinstance(value, dict):
            raise ValueError(
                "this needs to be a dictionary and you "
                "passed '%s' " % type(value))
        if ('clevel' not in value or
                'shuffle' not in value or
                'cname' not in value):
            raise ValueError(
                "The dictionary must have the next entries: "
                "'clevel', 'shuffle' and 'cname'")
        clevel, shuffle, cname = bcolz.cparams._checkparams(
            value['clevel'], value['shuffle'], value['cname'])
        return {'clevel': clevel, 'shuffle': shuffle, 'cname': cname}

    #
    # Properties start here...
    #

    @property
    def eval_vm(self):
        return self.__eval_vm

    @eval_vm.setter
    def eval_vm(self, value):
        self.check_choices('eval_vm', value)
        if value == "numexpr" and not bcolz.numexpr_here:
            raise (ValueError,
                   "cannot use `numexpr` virtual machine "
                   "(minimum required version is probably not installed)")
        self.__eval_vm = value

    @property
    def eval_out_flavor(self):
        return self.__eval_out_flavor

    @eval_out_flavor.setter
    def eval_out_flavor(self, value):
        self.check_choices('eval_out_flavor', value)
        self.__eval_out_flavor = value

    @property
    def cparams(self):
        return self.__cparams

    @cparams.setter
    def cparams(self, value):
        self.__cparams = self.check_cparams(value)


defaults = Defaults()


# Default values start here...

defaults.eval_out_flavor = "carray"
"""
The flavor for the output object in `eval()`.  It can be 'carray' or
'numpy'.  Default is 'carray'.
"""

defaults.eval_vm = "numexpr" if bcolz.numexpr_here else "python"
"""
The virtual machine to be used in computations (via `eval`).  It can
be 'numexpr' or 'python'.  Default is 'numexpr', if it is installed.
If not, then the default is 'python'.
"""

defaults.cparams = {'clevel': 5, 'shuffle': True, 'cname': 'blosclz'}
"""
The defaults for parameters used in compression.  You can change
them more comfortably by using the `cparams.setdefaults()` method.
"""
