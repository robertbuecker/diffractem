The Dataset object
==================

A diffractem data set is represented by a :class:`Dataset <diffractem.dataset.Dataset>` object, which manages all diffraction and meta data
from an electron diffraction set, and provides a plethora of features to work with them. This comprises:

* Automatic management of the HDF5 files containing the diffraction and meta data (see also 
  :ref:`file_format`).
* A framework to apply massively parallel computations on larger-than-memory diffraction data stacks
  using `dask <https://dask.org>`_, on a local machine or even remote clusters.
* Handling of meta data for each single recorded diffraction pattern using an embedded `pandas.DataFrame` 
  as a "shot list".
* Methods for quick and transparent creation of sub-sets through complex queries on metadata.

To learn how to handle `Dataset` objects, we'd recommend the `tutorials <https://github.com/robertbuecker/serialed-examples>`_.

Shot list
---------

Data stacks
-----------

Chunking
^^^^^^^^

Lazy evaluation and persisting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
