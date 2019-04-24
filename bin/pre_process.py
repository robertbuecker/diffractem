import matplotlib.pyplot as plt
import hdf5plugin

import lambda_tools.stream_parse
from lambda_tools import io, proc2d, tools, compute, overview
import numpy as np
from tifffile import TiffFile, imread, imsave
from glob import glob

file_mask = 'data2/DUT-8_S4_0??_00000.nxs'
list_name = 'procfiles.lst'
proc_folder = 'procdata2'
fig_folder = '/nas/photo/DUT-8/20190402/agg3'
min_peaks = 30
make_figures = True
reference_tif = 'Ref12_reference.tif'
pxmask_tif = 'Ref12_pxmask.tif'
raw_name = 'rawfiles.lst'

reference = imread(reference_tif)
pxmask = imread(pxmask_tif)
fns = glob(file_mask)
fns.sort()

with open(raw_name, 'w') as f: f.write('\n'.join(fns))

# Shot pre-selection and aggregation
sr0 = io.get_nxs_list(raw_name, 'shots')
sr0['selected'] = True
exclude = ['frame <= 0', 'frame >=3']
for ex in exclude:
    sr0.loc[sr0.eval(ex), 'selected'] = False
stack_raw, shots = io.modify_stack(sr0, drop_invalid=True,
                                   max_chunk=100, aggregate='sum', min_chunk=60,
                                   data_path='/entry/data/raw_counts')

# gymnastics required for paranoid data sets
crystals = io.get_nxs_list(raw_name, 'features')
shots = shots.drop(['crystal_x', 'crystal_y'], axis=1).merge(
    crystals[['crystal_x', 'crystal_y', 'crystal_id', 'subset']],
    on=['crystal_id', 'subset'], how='left')

io.copy_h5(raw_name, list_name, h5_folder=proc_folder, mode='w', exclude=('%/detector/data',))
io.store_nxs_list(list_name, shots, what='shots')

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
shots = io.get_nxs_list(list_name)
stacks = io.get_data_stacks(list_name, base_path='/%/data')

# mangle data arrays from computation into shot list
for key in ['adf1', 'adf2', 'beam_center', 'lorentz_fit']:
    data = stacks[key].compute()
    if data.ndim == 1:
        shots[key] = data
    else:
        for ii, col in enumerate(data.T):
            shots[key + '_' + '{}'.format(ii)] = col.T
io.store_nxs_list(list_name, shots)

# In[10]:
int_radius = (3, 4, 6)
crystfel_params = {'min-res': 0, 'max-res': 300,
                   'int-radius': '{},{},{}'.format(*int_radius), 'local-bg-radius': 3,
                   'threshold': 8, 'min-pix-count': 2, 'min-snr': 4,
                   'serial-start': 1, 'temp-dir': '/scratch/crystfel',
                   'peaks': 'peakfinder8', 'indexing': 'none'}

postfix = 'peaks'

stream_name = list_name.rsplit('.', 1)[0] + '_peaks.stream'
get_ipython().system(
    "{tools.call_indexamajig(list_name, 'parametric-sgl.geom', stream_name, im_params=crystfel_params)}")
peaks, predict = lambda_tools.stream_parse.read_crystfel_stream(stream_name)
shots = io.get_nxs_list(list_name, 'shots')
# hacks...
peaks = peaks.drop('subset', axis=1).merge(shots[['subset']], left_on='serial', right_index=True)
predict = peaks.drop('subset', axis=1).merge(shots[['subset']], left_on='serial', right_index=True)
print('{} peaks found in total.'.format(len(peaks)))

# Read crystfel stream file and weave the result into existing lists
pkdat = peaks.loc[:, ['serial', 'Intensity']].groupby('serial').agg(['count', 'sum', 'mean', 'median'])
pkdat.columns = ['peak_count', 'peak_int_sum', 'peak_int_mean', 'peak_int_median']
shots = shots[shots.columns.difference(pkdat.columns)].merge(pkdat, left_index=True, right_on='serial', how='left')
shots[pkdat.columns] = shots[pkdat.columns].fillna(0)
# meta.update({'shots': shots, 'peaks': peaks})
assert shots['peak_count'].sum() == len(peaks)
io.store_nxs_list(list_name, shots, what='shots')
io.store_nxs_list(list_name, peaks, what='peaks')
io.store_nxs_list(list_name, predict, what='predict')

if make_figures:

    niceones = shots.query(f'peak_count > {min_peaks}').index

    get_ipython().system('mkdir -r {folder}')
    for ii, dat in shots.query('peak_count > 25').iterrows():
        tools.diff_plot(list_name, [ii, ], map_px=17e-9, meta={'shots': shots, 'peaks': peaks},
                        clen=0.515, rings=(9, 4.5, 3.6, 2.25, 1.8))
        plt.savefig('{}/{:04d}_{:04d}'.format(folder, dat['region'], dat['shot']))
        plt.close(plt.gcf())
