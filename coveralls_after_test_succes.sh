pip install cpp-coveralls
coveralls --exclude c-blosc/internal-complibs --gcov-options '\-lp' --dump .coverage_cpp
pip uninstall -y cpp-coveralls
pip install coveralls
coveralls --merge=.coverage_cpp
