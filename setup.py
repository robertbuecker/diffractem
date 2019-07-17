from setuptools import setup

setup(
    name='diffractem',
    version='0.2.1',
    packages=['diffractem'],
    url='https://github.com/robertbuecker/diffractem',
    license='',
    scripts=['bin/nxs2tif.py', 'bin/edview.py'],
    author='Robert Buecker',
    author_email='robert.buecker@mpsd.mpg.de',
    description='Some tools for working with serial TEM data acquired with a Lambda Medipix-based detector.',
    install_requires=['h5py', 'numpy', 'pandas', 'tables', 'hdf5plugin',
                      'dask', 'tifffile', 'scipy', 'astropy', 'matplotlib', 'numba'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
