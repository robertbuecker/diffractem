import matplotlib.pyplot as plt
import hdf5plugin

import diffractem.dataset
import diffractem.stream_parser
from diffractem import io, proc2d, tools, compute, map_image
import numpy as np
from tifffile import TiffFile, imread, imsave
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser(description='Pre-processing for serial ED data.')

parser.add_argument('-i', '--input', type=str, help='nxs, h5, or lst input file.', required=True)
parser.add_argument('-o', '--output', type=str, help='output file, default: as input, moved to new folder.')
parser.add_argument('-p', '--pixel_mask', type=str, help='Pixel mask tif file.', required=True)
parser.add_argument('-d', '--dest_folder', type=str, help='Destination folder to store processed files.', default='procdata')
parser.add_argument('-f', '--flatfield', type=str, help='Flatfield image tif file.')
parser.add_argument('-c', '--centering', type=str, help='Type of centering algorithm.', choices=['none', 'com', 'fit'],
    default='fit')
parser.add_argument('-x' '--exclude', type=str, help='Exclusion string based on shot criteria. Can be used multiple times.',
    action='append')
parser.add_argument('-a' '--aggregate', type=str, help='Type of aggregation operation over shots.',
    choices=['none', 'frames', 'features'])
parser.add_argument('--keep-invalid', type=bool, action='store_true', help='Keep invalid (e.g. hysteresis reset) shots.')

args = parser.parse_args()

if args.flatfield is not None:
    reference = imread(args.flatfield)
else:
    reference = None

if args.pxmask is not None:
    pxmask = imread(args.pixel_mask)

raw_name = args.input

# Shot pre-selection and aggregation
sr0 = diffractem.dataset.get_nxs_list(raw_name, 'shots')
sr0['selected'] = True
for ex in args.exclude:
    sr0.loc[sr0.eval(ex), 'selected'] = False
stack_raw, shots = diffractem.dataset.modify_stack(sr0, drop_invalid=True,
                                                   max_chunk=100, aggregate='sum', min_chunk=60,
                                                   data_path='/entry/data/raw_counts')

# gymnastics required for paranoid data sets
crystals = diffractem.dataset.get_nxs_list(raw_name, 'features')
shots = shots.drop(['crystal_x', 'crystal_y'], axis=1).merge(
    crystals[['crystal_x', 'crystal_y', 'crystal_id', 'subset']],
    on=['crystal_id', 'subset'], how='left')

io.copy_h5(raw_name, list_name, h5_folder=proc_folder, mode='w', exclude=('%/detector/data',))
diffractem.dataset.store_nxs_list(list_name, shots, what='shots')

# Pre-processing: flatfield, dead pixels, center-of-mass, Lorentzian fit, centering, pixel mask creation

stack_ff = proc2d.apply_flatfield(stack_raw, reference)
stack = proc2d.correct_dead_pixels(stack_ff, pxmask, strategy='replace', replace_val=-1, mask_gaps=True)
stack_ct = proc2d.correct_dead_pixels(stack_ff[:, 0:516, 520:1036], pxmask[0:516, 520:1036],
                                      strategy='replace', replace_val=-1, mask_gaps=False)  # central sub-region
# invalid = da.all(stack_ct <= 0,axis=(1,2)) # bug due to occasional all-black images is fixed here:
thr = stack_ct.max(axis=1).topk(10, axis=1)[:, 9].reshape(
    (-1, 1, 1)) * 0.7  # kick out some of the highest pixels before defining the threshold
com = proc2d.center_of_mass2(stack_ct, threshold=thr) + [[520, 0]]
# com = da.where(invalid.reshape(-1,1), [[800, 250]], com)
lorentz = compute.map_reduction_func(stack, proc2d.lorentz_fast, com[:, 0], com[:, 1], radius=30, limit=26,
                                     scale=7, threads=False,
                                     output_len=4)  # Lorentz fit is only applied to region around found peak
ctr = lorentz[:, 1:3]
centered = proc2d.center_image(stack, ctr[:, 0], ctr[:, 1], 1556, 616, -1).astype(np.int16)
adf1 = proc2d.apply_virtual_detector(centered, 50, 100)
adf2 = proc2d.apply_virtual_detector(centered, 100, 200)

# all data to be stored in the final file
alldata = {'masked': stack, 'center_of_mass': com, 'lorentz_fit': lorentz, 'beam_center': ctr,
           'centered': centered.astype(np.int16),
           'pxmask_centered': (centered != -1).astype(np.uint16), 'adf1': adf1, 'adf2': adf2}

# This starts the actual crunching
io.store_data_stacks(list_name, alldata, flat=True, shots=shots, base_path='/%/data', compression=32004)

# take care of meta data
shots = diffractem.dataset.get_nxs_list(list_name)
stacks = io.get_data_stacks(list_name, base_path='/%/data')

# mangle data arrays from computation into shot list
for key in ['adf1', 'adf2', 'beam_center', 'lorentz_fit']:
    data = stacks[key].compute()
    if data.ndim == 1:
        shots[key] = data
    else:
        for ii, col in enumerate(data.T):
            shots[key + '_' + '{}'.format(ii)] = col.T
diffractem.dataset.store_nxs_list(list_name, shots)