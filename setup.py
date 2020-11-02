from setuptools import setup, Extension
import os

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
    version='0.3.3',
    packages=['diffractem'],
    url='https://github.com/robertbuecker/diffractem',
    license='',
    scripts=['bin/nxs2tif.py', 'bin/edview.py'],
    # scripts=['bin/nxs2tif.py', 'bin/edview.py', 'bin/quick_proc.py'],
    entry_points={
        'console_scripts': [
            'quick_proc = diffractem.quick_proc:main',
            'stream2sol = diffractem.stream_convert:main'
        ],
    },
    author='Robert Buecker',
    author_email='robert.buecker@cssb-hamburg.de',
    description='Some tools for working with serial electron microscopy data.',
    install_requires=['h5py', 'numpy', 'pandas', 'tables', 'hdf5plugin',
                      'dask[complete]', 'tifffile', 'scipy', 'astropy', 
                      'matplotlib', 'numba', 'pyqtgraph', 'pyyaml', 'scikit-learn', 
                      'scikit-image', 'PyQt5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    ext_modules = extensions
)
