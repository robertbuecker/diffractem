# diffractem

Pre-processing software for serial electron diffraction (SerialED) data.
See https://doi.org/10.1101/682575 for example results.

Diffractem is intended for usage within Jupyter notebooks - get a set of examples here: https://github.com/robertbuecker/serialed-examples.

## Installation
_diffractem_ is tailored to pre-processing SerialED data primarily for crystallographic analysis using _CrystFEL_, version 0.10.0 or higher: `https://www.desy.de/~twhite/crystfel/index.html`.
To make most of _diffractem_'s functionality, if you do not have it already, please download and install _CrystFEL_ following the installation instructions given on its homepage.
During the build process of _CrystFEL_ using _meson_, the _pinkIndexer_ component will automatically be downloaded and installed.

### Create conda enivronment
We _strongly_ suggest to use the Anaconda3 Python distribution/package manager, and create a dedicated environment within it for diffractem.
If you do not have Anaconda installed, it is sufficient to obtain the minimal _Miniconda_  of the `conda` package manager at https://docs.conda.io/en/latest/miniconda.

Once installed, please create a new anaconda environment for diffractem, and activate it:
```
conda create -n diffractem -c conda-forge python=3.10 numpy scipy pandas dask distributed jupyterlab ipywidgets ipympl tifffile h5py
conda activate diffractem
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
diffractem, (C) 2019-2022 Robert Bücker, robert.buecker@rigaku.com

peakfinder8, (C) 2014-2019 Deutsches Elektronen-Synchrotron DESY
