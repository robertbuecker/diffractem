# diffractem

Pre-processing software for serial electron diffraction data using Python.
See https://doi.org/10.1101/682575 for example results.

Diffractem is intended for usage within Jupyter notebooks - get a set of examples here: https://github.com/robertbuecker/serialed-examples.

Critical dependencies are _CrystFEL_ 0.10.0 or higher (http://www.desy.de/~twhite/crystfel/), available at https://gitlab.desy.de/thomas.white/crystfel, 
and _pinkIndexer_, available at https://stash.desy.de/users/gevorkov/repos/pinkindexer.

## Installation
To use most features you will also need an installation of _CrystFEL_ and _pinkIndexer_.

### Create conda enivronment
We _strongly_ suggest to use the Anaconda3 Python distribution/package manager, and create a dedicated environment within it for diffractem (and CrystFEL).
If you do not have Anaconda installed, it is sufficient to obtain the minimal _Miniconda_  of the `conda` package manager at https://docs.conda.io/en/latest/miniconda.

Once installed, please create a new anaconda environment for diffractem, **without installing any packages**, and activate it:
```
conda create -n diffractem -c conda-forge
conda activate diffractem
```
### Install CrystFEL and pinkIndexer
It is probably worth checking if you already have an installation of _CrystFEL_ (>=0.10.0) with _pinkIndexer_ in your computing environment (e.g. via _SBgrid_).
If so, you can skip this section and proceed with the installation of _diffractem_ below.
Otherwise, as you might not want the new CrystFEL version to interfere with a potentially existing (or future) installation of standard CrystFEL (or are  not allowed to), or don't have sudo rights on your computer, we suggest to install them directly into the location of our anaconda environment.

Please first make sure that all dependencies for compiling CrystFEL are installed:
https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/INSTALL.md

[N.B. If you're having trouble fulfilling CrystFELs dependencies e.g. because you do not have root access to install them or they are not included in your distribution's package repository, also consider installing them using `conda`.
As a common example, if you are using Ubuntu 16.04, which comes with a too old version of cmake, you may want to run `conda install cmake`.]

Here is a complete sequence of steps to install pinkIndexer and CrystFEL, assuming you're installing CrystFEL directly from the repository at DESY (alternatively, you can download the latest release at http://www.desy.de/~twhite/crystfel/):

```
git clone https://stash.desy.de/scm/~gevorkov/pinkindexer.git
git clone https://gitlab.desy.de/thomas.white/crystfel.git
cd pinkindexer
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j `nproc`
make install
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH"
cd ../../crystfel
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j `nproc`
make install
cd ../..
```

If you prefer to install CrystFEL globally, just omit the `-DCMAKE_INSTALL_PREFIX` parameters and skip the `export PKG_CONFIG_PATH` line.
After that, running `indexamajig --version` should return something similar to `CrystFEL: 0.10.0+<some commit hash>`.

### Install Python and Jupyter 
Now, install a basic Python with Jupyter into your environment.
We recommend using `jupyterlab` for interaction with diffractem.
If you prefer the classic `jupyter`, you can use it instead in the command below.
```
conda install -c conda-forge python=3.8 numpy jupyterlab ipywidgets ipympl
```
### Install diffractem

Finally install diffractem itself, either from PyPi:
```
pip install diffractem
```
or, if you want to play/develop a bit more and stay up-to-date, you can clone this git repository and install diffractem in developer mode:
```
git clone https://github.com/robertbuecker/diffractem
cd diffractem
pip install -e .
```


Now you should be ready to go! To get started, why don't you download the example notebooks:
```
git clone https://github.com/robertbuecker/serialed-examples
```
And get example raw data at MPDL Edmond: https://edmond.mpdl.mpg.de/imeji/collection/32lI6YJ7DZaF5L_K.

And when you're ready to go: just make your own branches of the notebooks for your own projects, and have fun!

---
diffractem, (C) 2019-2021 Robert Bücker, robert.buecker@cssb-hamburg.de

peakfinder8, (C) 2014-2019 Deutsches Elektronen-Synchrotron DESY
