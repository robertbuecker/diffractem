from distutils.core import setup

setup(
    name='lambda-tools',
    version='0.2.1',
    packages=['lambda_tools'],
    url='',
    license='',
    author='Robert Buecker',
    author_email='robert.buecker@mpsd.mpg.de',
    description='Some tools for working with serial TEM data acquired with a Lambda Medipix-based detector.',
    requires=['h5py','fabio','pytables','numpy','pandas','dask','tifffile','scipy','astropy']
)