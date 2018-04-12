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
import warnings

import numpy as np
import bcolz
from bcolz.py2help import xrange

if bcolz.numexpr_here:
    from numexpr.expressions import functions as numexpr_functions
if bcolz.dask_here:
    import dask.array as da


def is_sequence_like(var):
    "Check whether `var` looks like a sequence (strings are not included)."
    if hasattr(var, "__len__"):
        if isinstance(var, (bytes, str)):
            return False
        else:
            return True
    return False


def _getvars(expression, user_dict, vm):
    """Get the variables in `expression`."""

    cexpr = compile(expression, '<string>', 'eval')
    if vm in ("python", "dask"):
        exprvars = [var for var in cexpr.co_names
                    if var not in ['None', 'False', 'True']]
    else:
        # Check that var is not a numexpr function here.  This is useful for
        # detecting unbound variables in expressions.  This is not necessary
        # for the 'python' or 'dask' engines.
        exprvars = [var for var in cexpr.co_names
                    if var not in ['None', 'False', 'True']
                    and var not in numexpr_functions]

    # Get the local and global variable mappings of the user frame
    user_locals, user_globals = {}, {}
    user_frame = sys._getframe(2)
    user_locals = user_frame.f_locals
    user_globals = user_frame.f_globals

    # Look for the required variables
    reqvars = {}
    for var in exprvars:
        # Get the value
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
            hasattr(val, 'dtype') and is_sequence_like(val) and
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


def eval(expression, vm=None, out_flavor=None, user_dict={}, blen=None,
         **kwargs):
    """eval(expression, vm=None, out_flavor=None, user_dict=None, blen=None, **kwargs)

    Evaluate an `expression` and return the result.

    Parameters
    ----------
    expression : string
        A string forming an expression, like '2*a+3*b'. The values for 'a' and
        'b' are variable names to be taken from the calling function's frame.
        These variables may be scalars, carrays or NumPy arrays.
    vm : string
        The virtual machine to be used in computations.  It can be 'numexpr',
        'python' or 'dask'.  The default is to use 'numexpr' if it is
        installed.
    out_flavor : string
        The flavor for the `out` object.  It can be 'bcolz' or 'numpy'.
        If None, the value is get from `bcolz.defaults.out_flavor`.
    user_dict : dict
        An user-provided dictionary where the variables in expression
        can be found by name.
    blen : int
        The length of the block to be evaluated in one go internally.
        The default is a value that has been tested experimentally and
        that offers a good enough peformance / memory usage balance.
    kwargs : list of parameters or dictionary
        Any parameter supported by the carray constructor.

    Returns
    -------
    out : bcolz or numpy object
        The outcome of the expression.  In case out_flavor='bcolz',
        you can adjust the properties of this object by passing any
        additional arguments supported by the carray constructor in
        `kwargs`.

    """
    if vm is None:
        vm = bcolz.defaults.vm
    if vm not in ("numexpr", "python", "dask"):
        raise ValueError("`vm` must be either 'numexpr', 'python' or 'dask'")
    if vm == 'numexpr' and not bcolz.numexpr_here:
        raise ImportError("eval(..., vm='numexpr') requires numexpr, "
                          "which is not installed.")
    if vm == 'dask' and not bcolz.dask_here:
        raise ImportError("eval(..., vm='dask') requires dask, "
                          "which is not installed.")

    if out_flavor is None:
        out_flavor = bcolz.defaults.out_flavor

    # Get variables and column names participating in expression
    vars = _getvars(expression, user_dict, vm=vm)

    # Gather info about sizes and lengths
    typesize, vlen = 0, 1
    for name in vars:
        var = vars[name]
        if is_sequence_like(var) and not hasattr(var, "dtype"):
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
        if is_sequence_like(var):
            if vlen > 1 and vlen != len(var):
                raise ValueError("arrays must have the same length")
            vlen = len(var)

    if typesize == 0:
        # All scalars
        if vm in ("python", "dask"):
            return _eval(expression, vars)
        else:
            return bcolz.numexpr.evaluate(expression, local_dict=vars)

    return _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor, blen,
                        **kwargs)


def _eval_blocks(expression, vars, vlen, typesize, vm, out_flavor, blen,
                 **kwargs):
    """Perform the evaluation in blocks."""

    if not blen:
        # Compute the optimal block size (in elements)
        # The next is based on experiments with bench/ctable-query.py
        # and the 'movielens-bench' repository
        if vm == "numexpr":
            bsize = 2**23
        elif vm == "dask":
            bsize = 2**25
        else:  # python
            bsize = 2**21
        blen = int(bsize / typesize)
        # Protection against too large atomsizes
        if blen == 0:
            blen = 1

    if vm == "dask":
        if 'da' in vars:
            raise NameError(
                "'da' is reserved as a prefix for dask.array. "
                "Please use another prefix")
        for name in vars:
            var = vars[name]
            if is_sequence_like(var):
                vars[name] = da.from_array(var, chunks=(blen,) + var.shape[1:])
        # Build the expression graph
        vars['da'] = da
        da_expr = _eval(expression, vars)
        if out_flavor in ("bcolz", "carray") and da_expr.shape:
            result = bcolz.zeros(da_expr.shape, da_expr.dtype, **kwargs)
            # Store while compute expression graph
            da.store(da_expr, result)
            return result
        else:
            # Store while compute
            return np.array(da_expr)

    # Check whether we have a re_evaluate() function in numexpr
    re_evaluate = bcolz.numexpr_here and hasattr(bcolz.numexpr, "re_evaluate")

    vars_ = {}
    # Get containers for vars
    maxndims = 0
    for name in vars:
        var = vars[name]
        if is_sequence_like(var):
            ndims = len(var.shape) + len(var.dtype.shape)
            if ndims > maxndims:
                maxndims = ndims
            if len(var) > blen and hasattr(var, "_getrange"):
                    shape = (blen, ) + var.shape[1:]
                    vars_[name] = np.empty(shape, dtype=var.dtype)

    for i in xrange(0, vlen, blen):
        # Fill buffers for vars
        for name in vars:
            var = vars[name]
            if is_sequence_like(var) and len(var) > blen:
                if hasattr(var, "_getrange"):
                    if i+blen < vlen:
                        var._getrange(i, blen, vars_[name])
                    else:
                        vars_[name] = var[i:]
                else:
                    vars_[name] = var[i:i+blen]
            else:
                if np.isscalar(var):
                    vars_[name] = var
                elif hasattr(var, "__getitem__"):
                    vars_[name] = var[:]
                else:
                    vars_[name] = var

        # Perform the evaluation for this block
        if vm == "python":
            res_block = _eval(expression, vars_)
        else:
            if i == 0 or not re_evaluate:
                try:
                    res_block = bcolz.numexpr.evaluate(expression,
                                                       local_dict=vars_)
                except ValueError:
                    # numexpr cannot handle this, so fall back to "python" vm
                    warnings.warn(
                        "numexpr cannot handle this expression: falling back "
                        "to the 'python' virtual machine.  You can choose "
                        "another virtual machine by using the `vm` parameter.")
                    return _eval_blocks(
                        expression, vars, vlen, typesize, "python",
                        out_flavor, blen, **kwargs)
            else:
                res_block = bcolz.numexpr.re_evaluate(local_dict=vars_)

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
            if out_flavor in ("bcolz", "carray"):
                nrows = kwargs.pop('expectedlen', vlen)
                result = bcolz.carray(res_block, expectedlen=nrows, **kwargs)
            else:
                out_shape = list(res_block.shape)
                out_shape[0] = vlen
                result = np.empty(out_shape, dtype=res_block.dtype)
                result[:blen] = res_block
        else:
            if scalar or dim_reduction:
                result += res_block
            elif out_flavor in ("bcolz", "carray"):
                result.append(res_block)
            else:
                result[i:i+blen] = res_block

    if isinstance(result, bcolz.carray):
        result.flush()
    if scalar:
        return result[()]
    return result
