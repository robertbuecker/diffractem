# Ideas for diffractem features...

## Plotting

* high-level plot functions (in general) from older notebooks
* distribute in clever way between overview and dataset

## Maps

* map plot with unique axis projected length and orientation -> overview of e.g. preferred orientation.
Could be represented as complex image. Fill areas e.g. using watershed in intensity space. 
* lattice orientation clustering in maps 

## Viewer

* jump to arbitrary ID/serial
* real-/rec- space calibration, with diffraction rings and scale bar
* rudimentary keyboard operation
* line profiles,... check out glueviz?
* direct transfer to Fiji (check how scipion does it)

## Pre-processing

* function/script to automatically run full pre-proc pipeline, including some heuristics
* connect to running experiment, e.g. using socket interface or ZeroMQ

## Dataset

* direct CXI format deposition
* export to other formats (TIF, MRC, CBF, cctbx Pickle,...)

## Stream parser