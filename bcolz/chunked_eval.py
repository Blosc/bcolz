########################################################################
#
#       License: BSD
#       Created: July 15, 2014
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""Machinery for evaluating expressions.
"""

from __future__ import absolute_import

import sys
import math
import numpy as np
import bcolz
from bcolz.py2help import xrange

if bcolz.numexpr_here:
    from numexpr.expressions import functions as numexpr_functions


def _getvars(expression, user_dict, depth, vm):
    """Get the variables in `expression`.

    `depth` specifies the depth of the frame in order to reach local
    or global variables.
    """

    cexpr = compile(expression, '<string>', 'eval')
    if vm == "python":
        exprvars = [var for var in cexpr.co_names
                    if var not in ['None', 'False', 'True']]
    else:
        # Check that var is not a numexpr function here.  This is useful for
        # detecting unbound variables in expressions.  This is not necessary
        # for the 'python' engine.
        exprvars = [var for var in cexpr.co_names
                    if var not in ['None', 'False', 'True']
                    and var not in numexpr_functions]

    # Get the local and global variable mappings of the user frame
    user_locals, user_globals = {}, {}
    user_frame = sys._getframe(depth)
    user_locals = user_frame.f_locals
    user_globals = user_frame.f_globals

    # Look for the required variables
    reqvars = {}
    for var in exprvars:
        # Get the value.
        if var in user_dict:
            val = user_dict[var]
        elif var in user_locals:
            val = user_locals[var]
        elif var in user_globals:
            val = user_globals[var]
        else:
            if vm == "numexpr":
                raise NameError("variable name ``%s`` not found" % var)
            val = None
        # Check the value.
        if (vm == "numexpr" and
            hasattr(val, 'dtype') and hasattr(val, "__len__") and
                val.dtype.str[1:] == 'u8'):
            raise NotImplementedError(
                "variable ``%s`` refers to "
                "a 64-bit unsigned integer object, that is "
                "not yet supported in numexpr expressions; "
                "rather, use the 'python' vm." % var)
        if val is not None:
            reqvars[var] = val
    return reqvars


# Assign function `eval` to a variable because we are overriding it
_eval = eval


def eval(expression, vm=None, out_flavor=None, user_dict={}, **kwargs):
    """
    eval(expression, vm=None, out_flavor=None, user_dict=None, **kwargs)

    Evaluate an `expression` and return the result.

    Parameters
    ----------
    expression : string
        A string forming an expression, like '2*a+3*b'. The values for 'a' and
        'b' are variable names to be taken from the calling function's frame.
        These variables may be scalars, carrays or NumPy arrays.
    vm : string
        The virtual machine to be used in computations.  It can be 'numexpr'
        or 'python'.  The default is to use 'numexpr' if it is installed.
    out_flavor : string
        The flavor for the `out` object.  It can be 'carray' or 'numpy'.
    user_dict : dict
        An user-provided dictionary where the variables in expression
        can be found by name.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : carray object
        The outcome of the expression.  You can tailor the
        properties of this carray by passing additional arguments
        supported by carray constructor in `kwargs`.

    """
    if vm is None:
        vm = bcolz.defaults.eval_vm
    if vm not in ("numexpr", "python"):
        raise ValueError("`vm` must be either 'numexpr' or 'python'")

    if out_flavor is None:
        out_flavor = bcolz.defaults.eval_out_flavor
    if out_flavor not in ("carray", "numpy"):
        raise ValueError("`out_flavor` must be either 'carray' or 'numpy'")

    # Get variables and column names participating in expression
    depth = kwargs.pop('depth', 2)
    vars = _getvars(expression, user_dict, depth, vm=vm)

    # Gather info about sizes and lengths
    typesize, vlen = 0, 1
    for name in vars:
        var = vars[name]
        if hasattr(var, "__len__") and not hasattr(var, "dtype"):
            raise ValueError("only numpy/carray sequences supported")
        if hasattr(var, "dtype") and not hasattr(var, "__len__"):
            continue
        if hasattr(var, "dtype"):  # numpy/carray arrays
            if isinstance(var, np.ndarray):  # numpy array
                typesize += var.dtype.itemsize * np.prod(var.shape[1:])
            elif isinstance(var, bcolz.carray):  # carray array
                typesize += var.dtype.itemsize
            else:
                raise ValueError("only numpy/carray objects supported")
        if hasattr(var, "__len__"):
            if vlen > 1 and vlen != len(var):
                raise ValueError("arrays must have the same length")
            vlen = len(var)

    if typesize == 0:
        # All scalars
        if vm == "python":
            return _eval(expression, vars)
        else:
            return bcolz.numexpr.evaluate(expression, local_dict=vars)

    return _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
                        **kwargs)


def _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor,
                 **kwargs):
    """Perform the evaluation in blocks."""

    # Compute the optimal block size (in elements)
    # The next is based on experiments with bench/ctable-query.py
    # and the 'movielens-bench' repository
    if vm == "numexpr":
        bsize = 2**24
    else:
        bsize = 2**22
    bsize //= typesize
    # Evaluation seems more efficient if block size is a power of 2
    bsize = 2 ** (int(math.log(bsize, 2)))
    if vlen < 100*1000:
        bsize //= 8
    elif vlen < 1000*1000:
        bsize //= 4
    elif vlen < 10*1000*1000:
        bsize //= 2
    # Protection against too large atomsizes
    if bsize == 0:
        bsize = 1

    vars_ = {}
    # Get temporaries for vars
    maxndims = 0
    for name in vars:
        var = vars[name]
        if hasattr(var, "__len__"):
            ndims = len(var.shape) + len(var.dtype.shape)
            if ndims > maxndims:
                maxndims = ndims
            if len(var) > bsize and hasattr(var, "_getrange"):
                vars_[name] = np.empty(bsize, dtype=var.dtype)

    for i in xrange(0, vlen, bsize):
        # Get buffers for vars
        for name in vars:
            var = vars[name]
            if hasattr(var, "__len__") and len(var) > bsize:
                if hasattr(var, "_getrange"):
                    if i+bsize < vlen:
                        var._getrange(i, bsize, vars_[name])
                    else:
                        vars_[name] = var[i:]
                else:
                    vars_[name] = var[i:i+bsize]
            else:
                if hasattr(var, "__getitem__"):
                    vars_[name] = var[:]
                else:
                    vars_[name] = var

        # Perform the evaluation for this block
        if vm == "python":
            res_block = _eval(expression, vars_)
        else:
            try:
                res_block = bcolz.numexpr.evaluate(expression,
                                                   local_dict=vars_)
            except ValueError:
                # numexpr cannot handle this. Fall back to a pure "python" VM.
                return _eval_blocks(
                    expression, vars, vlen, typesize, "python",
                    out_flavor, **kwargs)

        if i == 0:
            # Detection of reduction operations
            scalar = False
            dim_reduction = False
            if len(res_block.shape) == 0:
                scalar = True
                result = res_block
                continue
            elif len(res_block.shape) < maxndims:
                dim_reduction = True
                result = res_block
                continue
            # Get a decent default for expectedlen
            if out_flavor == "carray":
                nrows = kwargs.pop('expectedlen', vlen)
                result = bcolz.carray(res_block, expectedlen=nrows, **kwargs)
            else:
                out_shape = list(res_block.shape)
                out_shape[0] = vlen
                result = np.empty(out_shape, dtype=res_block.dtype)
                result[:bsize] = res_block
        else:
            if scalar or dim_reduction:
                result += res_block
            elif out_flavor == "carray":
                result.append(res_block)
            else:
                result[i:i+bsize] = res_block

    if isinstance(result, bcolz.carray):
        result.flush()
    if scalar:
        return result[()]
    return result
