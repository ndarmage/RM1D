"""from https://numpy.org/doc/stable/f2py/buildtools/distutils.html"""
from pathlib import Path

from numpy.distutils.core import Extension, setup
from setuptools import find_packages


src_dir = Path('src') # Path allows to be complient wit Linux and Windows
f_flags = [ # from CPM1D/tools/wrap_f77_to_pylib.sh
           '-ffixed-line-length-72',
           '-Wextra',
           '-Wall',
           '-std=legacy'
          ]
extension = Extension(name='KinPy.algo609',
                      sources=[str(src_dir / 'KinPy' / 'algo609.f')],
                      # see https://numpy.org/doc/stable/reference/generated/numpy.distutils.core.Extension.html#numpy.distutils.core.Extension
                      # for compile options.
                      extra_f77_compile_args=f_flags,
                      extra_f90_compile_args=f_flags,
                      )

setup(name='CPM1D',
      description=("Services to compute the 1D integral transport"
                   "equation by the collision probability method"),
      author="Daniele tomatis",
      author_email="???",
      version="0.0.1",
      ext_modules=[extension,],
      install_requires = ["setuptools", "wheel", "numpy"],
      package_dir={"": "src"},  # To install KinPy and all python files in it (requires __init__.py in each subfolders to install)
      packages=find_packages(where="src"),  # to locate src dir
      )
