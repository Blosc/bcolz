#!/bin/bash
printf "\nConfig\n"
echo $CONDA_BUILD_SYSROOT
echo $MACOSX_DEPLOYMENT_TARGET
printf "\nConfig\n"

python -m pip install -v --install-option="-Isysroot=${CONDA_BUILD_SYSROOT}" .
