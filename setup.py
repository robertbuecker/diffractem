from setuptools import setup, Extension
import os

# DIFFRACTEM - tools for processing Serial Electron Diffraction Data
# Copyright (C) 2020  Robert Bücker

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

# This library uses peakfinder8 for peak finding, written by Anton Barty,
# Valerio Mariani, and Oleksandr Yefanov;
# Copyright 2014-2019 Deutsches Elektronen-Synchrotron DESY


### ---
# peakfinder8 Cython version adapted from OnDA: https://github.com/ondateam/onda

try:
    import numpy
except (ModuleNotFoundError, NameError):
    print('NumPy is not installed. Please install it before diffractem via:\n'
          'pip install numpy')


DIFFRACTEM_USE_CYTHON = os.getenv("DIFFRACTEM_USE_CYTHON")

ext = ".pyx" if DIFFRACTEM_USE_CYTHON else ".c"  # pylint: disable=invalid-name

peakfinder8_ext = Extension(  # pylint: disable=invalid-name
    name="diffractem.peakfinder8_extension",
    include_dirs=[numpy.get_include()],
    libraries=["stdc++"],
    sources=[
        "src/peakfinder8_extension/peakfinder8.cpp",
        "src/peakfinder8_extension/peakfinder8_extension.pyx",
    ]
    if DIFFRACTEM_USE_CYTHON
    else [
        "src/peakfinder8_extension/peakfinder8_extension.cpp",
        "src/peakfinder8_extension/peakfinder8.cpp",
    ],
    language="c++",
)

if DIFFRACTEM_USE_CYTHON:
    from Cython.Build import cythonize
    print('USING CYTHON')
    extensions = cythonize(peakfinder8_ext)  # pylint: disable=invalid-name
else:
    extensions = [peakfinder8_ext]  # pylint: disable=invalid-name
    
### ---

setup(
    name='diffractem',
    version='0.4.0b1',
    packages=['diffractem'],
    url='https://github.com/robertbuecker/diffractem',
    license='',
    scripts=['bin/nxs2tif.py', 'bin/edview.py'],
    # scripts=['bin/nxs2tif.py', 'bin/edview.py', 'bin/quick_proc.py'],
    entry_points={
        'console_scripts': [
            'quick_proc = diffractem.quick_proc:main',
            'stream2sol = diffractem.stream2sol:main'
        ],
    },
    author='Robert Buecker',
    author_email='robert.buecker@cssb-hamburg.de',
    description='Some tools for working with serial electron microscopy data.',
    install_requires=['h5py', 'numpy', 'pandas', 'hdf5plugin',
                      'dask[complete]', 'tifffile', 'scipy', 'astropy', 
                      'matplotlib', 'numba', 'pyqtgraph', 'pyyaml', 'scikit-learn', 
                      'scikit-image', 'PyQt5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
    ],
    ext_modules = extensions,
    include_package_data = True
)
