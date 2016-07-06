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

from contextlib import contextmanager
import bcolz


class Defaults(object):
    """Class to taylor the setters and getters of default values."""

    def __init__(self):
        self.choices = {}

        # Choices setup
        self.choices['out_flavor'] = ("bcolz", "carray", "numpy")
        self.choices['vm'] = ("numexpr", "python", "dask")

    def check_choices(self, name, value):
        if value not in self.choices[name]:
            raise ValueError(
                "'%s' is incorrect value for '%s' default" % (value, name))

    def check_cparams(self, value):
        entries = ['clevel', 'shuffle', 'cname', 'quantize']
        if isinstance(value, bcolz.cparams):
            value = dict((e, getattr(value, e)) for e in entries)
        if not isinstance(value, dict):
            raise ValueError(
                "this needs to be a cparams object or a dictionary and you "
                "passed '%s' " % type(value))
        if not all(k in value for k in entries):
            raise ValueError(
                "The dictionary must have the next entries: %s" % entries)
        # Return a dictionary with the proper defaults
        return dict(zip(entries, bcolz.cparams._checkparams(**value)))

    #
    # Properties start here...
    #

    @property
    def vm(self):
        return self.__vm

    @vm.setter
    def vm(self, value):
        self.check_choices('vm', value)
        if value == "numexpr" and not bcolz.numexpr_here:
            raise ValueError(
                "cannot use `numexpr` virtual machine "
                "(minimum required version is probably not installed)")
        elif value == "dask" and not bcolz.dask_here:
            raise ValueError(
                "cannot use `dask` virtual machine "
                "(minimum required version is probably not installed)")
        self.__vm = value

    # Keep eval_vm for backward compatibility
    eval_vm = vm

    @property
    def out_flavor(self):
        return self.__out_flavor

    @out_flavor.setter
    def out_flavor(self, value):
        self.check_choices('out_flavor', value)
        self.__out_flavor = value

    # Keep eval_out_flavor for backward compatibility
    eval_out_flavor = out_flavor

    @property
    def cparams(self):
        return self.__cparams

    @cparams.setter
    def cparams(self, value):
        self.__cparams = self.check_cparams(value)


defaults = Defaults()


# Default values start here...

defaults.out_flavor = "bcolz"
"""The flavor for the output object in `eval()`.  It can be 'bcolz'
or 'numpy'.  Default is 'bcolz'.

"""

if bcolz.numexpr_here:
    defaults.vm = "numexpr"
elif bcolz.dask_here:
    defaults.vm = "dask"
else:
    defaults.vm = "python"
"""The virtual machine to be used in computations (via `eval`).  It
can be 'numexpr', 'dask' or 'python'.  Default is 'numexpr', if it is
installed.  If not, 'dask' is used, if installed.  And if neither of
these are installed, then the 'python' interpreter is used.

"""

defaults.cparams = {'clevel': 5, 'shuffle': bcolz.SHUFFLE,
                    'cname': 'lz4', 'quantize': 0}
"""The defaults for parameters used in compression.  You can change
them more comfortably by using the `cparams.setdefaults()` method.

"""


@contextmanager
def defaults_ctx(cparams=None, vm=None, out_flavor=None):
    """Execute a context with some defaults"""
    cparams_orig, vm_orig, out_flavor_orig = None, None, None
    if cparams:
        cparams_orig = defaults.cparams
        defaults.cparams = cparams
    if vm:
        vm_orig = defaults.vm
        defaults.vm = vm
    if out_flavor:
        out_flavor_orig = defaults.out_flavor
        defaults.out_flavor = out_flavor

    yield

    if cparams_orig:
        defaults.cparams = cparams_orig
    if vm_orig:
        defaults.vm = vm_orig
    if out_flavor_orig:
        defaults.out_flavor = out_flavor_orig
