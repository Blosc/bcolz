
def build_carray(rootdir):
    from bcolz import carray
    return carray(rootdir=rootdir)

def build_ctable(rootdir):
    from bcolz import ctable
    return ctable(rootdir=rootdir)
