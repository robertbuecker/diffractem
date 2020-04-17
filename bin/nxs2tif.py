#!/usr/bin/env python
from tifffile import imsave
import h5py
import sys
import numpy as np

fh = h5py.File(sys.argv[1])
ds = fh['/entry/instrument/detector/data']
if len(sys.argv) > 2:
    fn = sys.argv[2]
else:
    fn = sys.argv[1].rsplit('.', 1)[0] + '.tif'
if ds.dtype == np.int32:
    ds = ds[:].astype(np.float32)
imsave(fn, ds[:, :, :])
print('Wrote ' + fn)