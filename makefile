# This is the maintainers undocumented Makefile.
# Nothin to see here, please move along.
.PHONY: build test clean docs

build:
	python setup.py build_ext -i

test:
	python -c "import bcolz ; bcolz.test(heavy=True)"

clean:
	git clean -dfX; git clean -dfx

docs:
	cd docs && make html
