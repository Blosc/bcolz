#!/bin/bash
echo $CONDA_BUILD_SYSROOT
echo $MACOSX_DEPLOYMENT_TARGET
export CFLAGS="${CFLAGS} -i sysroot ${CONDA_BUILD_SYSROOT}"
echo $CFLAGS

python setup.py install -v
