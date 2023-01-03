#!/usr/bin/env python
# encoding: UTF-8


from setuptools import setup, find_packages, dist

"""
# Declare the dependency
dist.Distribution(dict(setup_requires='pythran'))
"""

# Pythran modules
try:
    from pythran import PythranExtension

    module_interpolation = PythranExtension(
        "ECOdyagnostics.interpolation_compiled", sources=["ECOdyagnostics/_interpolation.py"]
    )
    EXT_MODULES = [module_interpolation]
except ModuleNotFoundError:
    # Compilation is not possible
    EXT_MODULES = []

DISTNAME = "ECOdiagnostics"
VERSION = "0.0.0"
LICENSE = "MIT"
AUTHOR = "Benjamin Schmiedel"
AUTHOR_EMAIL = "benny.schmiedel@gmail.com"
URL = ""
CLASSIFIERS = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
]
INSTALL_REQUIRES = ["xarray", "dask", "numpy", "pythran>=0.9.5", "gsw", "xnemogcm", "configparser", ]
DESCRIPTION = "A diagnostics package for the Energy Cycle of the Ocean"
PYTHON_REQUIRES = ">=3.6"


def readme():
    with open("README.md") as f:
        return f.read()


# Main setup:
setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    long_description=readme(),
    url=URL,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires=PYTHON_REQUIRES,
    ext_modules=EXT_MODULES,
    include_package_data=True,
)
