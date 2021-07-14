.. diffractem documentation master file, created by
   sphinx-quickstart on Fri Apr 24 17:13:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to diffractem's documentation!
======================================

diffractem is a package for processing Serial Electron Diffraction data, following the protocols as outlined in `Bücker et al., Front. Mol. Biosci., 2021 <https://doi.org/10.3389/fmolb.2021.624264>`_.
See this paper for a general introduction and documentation
diffractem is mostly intended to be used from within Jupyter notebooks, such as those available from `here <https://github.com/robertbuecker/serialed-examples>`_.

Please see :ref:`diffractem:Installation` for how to install diffractem and CrystFEL such that you can get started.

Of particular interest might be the documentation of :class:`PreProcOpts <diffractem.pre_proc_opts.PreProcOpts>`, which explains the various options you can define for preprocessing.

For the full  API documentation, see :ref:`here <diffractem:Submodules>`

Table of contents
-----------------

.. toctree::
   :maxdepth: 4

   Overview<diffractem>
   dataset
   file_format
   edview
   CrystFEL integration<crystfel>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
