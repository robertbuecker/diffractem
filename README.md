# diffractem

Pre-processing software for serial electron diffraction data using Python.
See https://doi.org/10.1101/682575 for example results.

Diffractem is intended for usage within Jupyter notebooks - get a set of examples here: https://github.com/robertbuecker/serialed-examples.

Critical dependencies are the electron-enabled version of _CrystFEL_ (http://www.desy.de/~twhite/crystfel/), available at https://stash.desy.de/projects/MPSDED/repos/crystfel/, 
and _pinkIndexer_, available at https://stash.desy.de/users/gevorkov/repos/pinkindexer.

## Installation
...in three steps:
* set up a python environment
* install diffractem itself
* install _CrystFEL_ and _pinkIndexer_

### Set up Python (conda) environment 

To install diffractem, we _strongly_ suggest to use the Anaconda3 Python distribution, and create a dedicated environment within it.
If you do not have Anaconda yet, please get either the minimal Miniconda version at or the minimal version at https://docs.conda.io/en/latest/miniconda.html, or the full version at https://www.anaconda.com/products/individual.

Once installed, please create a new anaconda environment for diffractem:
```
conda create -n diffractem -c conda-forge python=3.8 numpy ipykernel
```
You can append more packages at your liking. 
E.g. if you do not have a Jupyter installation on your system yet (check by running `jupyter notebook` and see if something happens), you might want to add `jupyter`, or even `jupyterlab` (more modern version).

### Install diffractem

Next, activate your new environment
```
conda activate diffractem
```
...and install diffractem itself, either from PyPi:
```
pip install diffractem
```
or, if you want to play/develop a bit more and stay up-to-date, you can clone this git repository and install diffractem in developer mode:
```
git clone https://github.com/robertbuecker/diffractem
cd diffractem
pip install -e .
```

### Install CrystFEL and pinkIndexer
Please first make sure that all dependcies for compiling CrystFEL are installed, following Steps 1-3 here (but NOT Step 4 - it is the wrong CrystFEL version):
https://www.desy.de/~twhite/crystfel/install.html

As you might not want the electron-enabled CrystFEL version to interfere with a potentially existing (or future) installation of standard CrystFEL, we suggest to install it into the `conda` environment we created above.
This way, the electron CrystFEL version will be enabled (put on the path first) as soon as the environment is activated.

We start with _pinkIndexer_ (otherwise CrystFEL will not recognize its presence):
```
conda activate diffractem
git clone https://stash.desy.de/users/gevorkov/repos/pinkindexer
cd pinkindexer
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
```

And finally _CrystFEL_ (electron version) itself:
```
conda activate diffractem
git clone https://stash.desy.de/projects/MPSDED/repos/crystfel
cd crystfel
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make
make install
```

Now you should be ready to go! To get started, why don't you download the example notebooks:
```
git clone https://github.com/robertbuecker/serialed-examples
```
And get example raw data at www.empiar.org under the ID 10542.

And when you're ready to go: just make your own branches of the notebooks for your own projects, and have fun!

---
diffractem, (C) 2020 Robert Bücker, robert.buecker@cssb-hamburg.de

peakfinder8, (C) 2014-2019 Deutsches Elektronen-Synchrotron DESY