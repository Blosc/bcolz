#!/bin/bash
printf "\nConfig\n"
echo $CONDA_BUILD_SYSROOT
echo $MACOSX_DEPLOYMENT_TARGET
conda list
which python
printf "\nConfig\n"

python -m pip install -vv .