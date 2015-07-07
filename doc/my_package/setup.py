from setuptools import setup, Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs


# Sources
sources = ["my_extension/example_ext.pyx"]

setup(
    name="my_package",
    description='My description',
    license='MY_LICENSE', 
    ext_modules=[
        Extension(
            "my_extension.example_ext",
            sources=sources,
        ),
    ],
    cmdclass={"build_ext": build_ext},
    packages=['my_extension'],
)
