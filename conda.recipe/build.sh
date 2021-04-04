#!/bin/bash
# install using pip from the whl provided by PyPI

if [ $(uname) == Darwin ]; then
  if [ "$PY_VER" == "3.7" ]; then
    pip install https://pypi.io/packages/source/b/bcolz-zipline/bcolz_zipline-1.2.3-cp37-cp37m-macosx_10_9_x86_64.whl
  else
    if [ "$PY_VER" == "3.8" ]; then
      pip install https://pypi.io/packages/source/b/bcolz-zipline/bcolz_zipline-1.2.3-cp38-cp38m-macosx_10_9_x86_64.whl
    else
      pip install https://pypi.io/packages/source/b/bcolz-zipline/bcolz_zipline-1.2.3-cp39-cp39m-macosx_10_9_x86_64.whl
    fi
  fi
fi
