from distutils.core import setup

setup(
    name='lambda-tools',
    version='0.2.1',
    packages=['diffractem'],
    url='',
    license='',
    scripts=['bin/nxs2tif.py','bin/streamViewer.py','bin/sedViewer.py'],
    author='Robert Buecker',
    author_email='robert.buecker@mpsd.mpg.de',
    description='Some tools for working with serial TEM data acquired with a Lambda Medipix-based detector.',
    #install_requires=['h5py','fabio','numpy','pandas',
    #          'dask','tifffile','scipy','astropy','matplotlib','scikit-learn','numba']
)
