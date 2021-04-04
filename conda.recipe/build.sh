#!/bin/bash
printf "\nConfig\n"
echo $CONDA_BUILD_SYSROOT
echo $MACOSX_DEPLOYMENT_TARGET
export CFLAGS="${CFLAGS} --sysroot=${CONDA_BUILD_SYSROOT}"
echo $CFLAGS
printf "\nConfig\n"

python setup.py install -v
