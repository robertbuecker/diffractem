# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
import hdf5plugin # required to access LZ4-encoded HDF5 data sets
import matplotlib.pyplot as plt
from diffractem import io, tools, proc_peaks, version, proc2d, compute
from diffractem.dataset import Dataset
from diffractem.stream_parser import StreamParser
from diffractem import pre_process
from tifffile import imread
import numpy as np
import pandas as pd
import os
import dask.array as da
import h5py
from dask.distributed import Client, LocalCluster

opts = pre_process.PreProcOpts('preproc_multiproc.yaml')
# opts.im_exc = 'indexamajig'
cfver = get_ipython().getoutput('{opts.im_exc} -v')
print(f'Running on diffractem:', version())
print(f'Running on', cfver[0])
print(f'Current path is:', os.getcwd())

# %% [markdown]
# # dask.distributed

# %%
# from dask.distributed import Client
cluster = LocalCluster(n_workers=20, threads_per_worker=1)
client = Client(cluster)


# %%
cluster


# %%
opts.load()

raw_files = io.expand_files('data/*.nxs')
print(f'Have {len(raw_files)} raw files. Have fun pre-processing!')


# %%
# load raw files
dsraw = Dataset.from_list(raw_files, init_stacks=False)
# %mkdir proc_data


# %%
# "atomic aggregate" -> refactor into diffractem!
def get_agg_img(file, chunk=40):
    dsraw_sgl = Dataset.from_list(file, init_stacks=False)
    dsraw_sgl.open_stacks(readonly=True, swmr=False, chunking=chunk)
    dsagg_sgl = dsraw_sgl.aggregate(query=opts.agg_query, new_folder=opts.proc_dir, 
                                    file_suffix=opts.agg_file_suffix,how={'raw_counts': 'sum'})
    return dsagg_sgl.raw_counts, dsagg_sgl.shots, dsraw_sgl

# TODO: can this (dask graph setup) be parallelized? Probably...
allagg = [get_agg_img(fn) for fn in dsraw.files[:1]]

raw1 = allagg[0][2]
raw1.raw_counts

rawsel = raw1.get_selection(query='frame != -1')


# %%
raw1.raw_counts[:20,...].visualize(optimize_graph=True)


# %%
rawsel.raw_counts[:20].visualize(optimize_graph=True)


# %%
rawsel.shots.head(20)


# %%
for (reg, run, cid), grp in rawsel.shots.groupby(['region','run','crystal_id'], sort=False):
    print(cid, grp.index[0])


# %%
ds = Dataset.from_list(raw_files[0])
ds.open_stacks(chunking=20, readonly=True, swmr=False)
rawsel = ds.get_selection('frame >= 0')


# %%
get_ipython().run_cell_magic('time', '', "rawagg = rawsel.aggregate(how='sum', force_commensurate=True)")


# %%
rawagg.raw_counts[:5,...].visualize(optimize_graph=True)


# %%
agg(rawsel).shots


# %%
agg(rawsel, how='sum', query='frame == 0').shots


# %%
agg(rawsel, how='sum', query='frame == 0').raw_counts[:8,...].visualize(optimize_graph=True)


# %%
ds = allagg[0][2]._h5handles['data/GV_S11_001_00000.nxs']['/entry/data/raw_counts']


# %%
allshot = pd.concat([a[1] for a in allagg], axis=0)
allimg = da.concatenate([a[0] for a in allagg], axis=0)


# %%
# generate agg set indirecly
dsraw.open_stacks(readonly=True, swmr=False)
dsraw.delete_stack('raw_counts', from_files=False)
# check why this takes so long! Also, check if it is required at all...
dsagg = dsraw.aggregate(query=opts.agg_query, new_folder=opts.proc_dir, 
                                    file_suffix=opts.agg_file_suffix)
assert (dsagg.shots[['file', 'Event']] == allshot.reset_index()[['file', 'Event']]).all().all()
dsagg.add_stack('raw_counts', allimg)


# %%
assert dsagg.raw_counts[:5,...].compute(scheduler=client.get).shape == (5,516,1556)


# %%
# %%time
# # this builds the dask graph for aggregation, and is unfortunately single-threaded and slow

# # dssel = dsraw.get_selection('frame > 0 and frame < 5', file_suffix='.h5')
# dsagg = dsraw.aggregate(query=opts.agg_query, new_folder=opts.proc_dir, file_suffix=opts.agg_file_suffix)


# %%
pxmask=imread(opts.pxmask)
reference=imread(opts.reference)


# %%
# sort-of-conventional pipeline
from diffractem.proc_peaks import _ctr_from_pks
from diffractem.proc2d import center_of_mass, lorentz_fast, loop_over_stack

stack = proc2d.apply_flatfield(dsagg.raw_counts.rechunk((20,-1,-1)), imread(opts.reference), keep_type=False)
stack = proc2d.correct_dead_pixels(stack, imread(opts.pxmask), strategy='replace', replace_val=-1, mask_gaps=False)

pkmax = 500

@loop_over_stack
def get_center(img):
    # consider refactoring this into lorentz_fast if called without initial values
    img_ct = img[:,(img.shape[1]-opts.com_xrng)//2:(img.shape[1]+opts.com_xrng)//2]
    amp = np.quantile(img_ct,1-5e-5)
    com = center_of_mass(img_ct, threshold=amp*0.7) + [(img.shape[1]-opts.com_xrng)//2, 0]
    print(com)
    lorentz = lorentz_fast(img, com[0], com[1], radius=opts.lorentz_radius,
                                         limit=opts.lorentz_maxshift, scale=7, threads=False)
    return np.concatenate((com, lorentz))

ctr_data = compute.map_reduction_func(stack, get_center, output_len=6)
peak_data = compute.map_reduction_func(stack, proc2d.get_peaks, ctr_data[:,2], ctr_data[:,3],
                                      max_peaks=pkmax, pxmask=pxmask, output_len=3*pkmax+1)

radial_prof = None

@loop_over_stack
def refine_center(img, ctr0, pk_x, pk_y, pk_I, npk, 
                  maxres=None, int_weight=False,
                  sigma=2.0, bound=5.0):
    # we use a custom function instead of proc_peaks.center_friedel, in order to 
    
    pkl = np.stack((pk_x, pk_y, pk_I), -1)[:npk,:]
    if maxres is not None:
        rsq = (pkl[:, 0] - ctr0[0]) ** 2 + (pkl[:, 1] - ctr0[1]) ** 2
        pkl = pkl[rsq < maxres ** 2, :]
    ctr_refined, cost, _ = _ctr_from_pks(pkl, ctr0, int_weight=int_weight, 
                  sigma=sigma, bound=bound)
    
    return np.hstack((ctr_refined, cost))


# %%
# single-loop pipeline (might be faster)
from diffractem.proc_peaks import _ctr_from_pks

@proc2d.loop_over_stack
def get_pattern_info(img, opts, reference, pxmask):
    
    # apply flatfield and dead-pixel correction to get more accurate COM
    # CONSIDER DOING THIS OUTSIDE GET PATTERN INFO!
    img = proc2d.apply_flatfield(img, reference, keep_type=False)
    img = proc2d.correct_dead_pixels(img, pxmask, strategy='replace', mask_gaps=False, replace_val=-1)
    
    # thresholded center-of-mass calculation over x-axis sub-range
    img_ct = img[:,(img.shape[1]-opts.com_xrng)//2:(img.shape[1]+opts.com_xrng)//2]
    com = proc2d.center_of_mass(img_ct, threshold=opts.com_threshold*np.quantile(img_ct,1-5e-5)) + [(img.shape[1]-opts.com_xrng)//2, 0]
    
    # Lorentz fit of direct beam
    lorentz = proc2d.lorentz_fast(img, com[0], com[1], radius=opts.lorentz_radius,
                                         limit=opts.lorentz_maxshift, scale=7, threads=False)
    
    # Get peaks using peakfinder8. Note that pf8 parameters are taken straight from the options file,
    # with automatic underscore/hyphen handling.
    peak_data = proc2d.get_peaks(img, lorentz[1], lorentz[2], pxmask=pxmask, max_peaks=opts.max_peaks,
                                **{k.replace('-','_'): v for k, v in opts.peak_search_params.items()})
    
    # Refine center using Friedel-mate matching
    if peak_data[-1] >= opts.min_peaks:  
        
        # prepare peak list. Note the .5, as _ctr_from_pks expects CrystFEL peak convention
        pkl = np.stack((peak_data[:opts.max_peaks] + .5, peak_data[opts.max_peaks:2*opts.max_peaks] + .5, 
                        peak_data[2*opts.max_peaks:-1]), -1)[:int(peak_data[-1]),:]
        if opts.friedel_max_radius is not None:
            rsq = (pkl[:, 0] - ctr0[0]) ** 2 + (pkl[:, 1] - ctr0[1]) ** 2
            pkl = pkl[rsq < maxres ** 2, :]
        
        ctr_refined, cost, _ = _ctr_from_pks(pkl, lorentz[1:3], int_weight=False, 
                      sigma=opts.peak_sigma)
        
    else:
        ctr_refined, cost = lorentz[1:3], np.nan

    return np.concatenate((com, lorentz, peak_data, ctr_refined, cost), axis=None)

def unravel_pattern_info(info, opts, shots=None):
    # Function to grab result matrix from get_pattern_info and convert it to
    # a pandas DataFrame and a CXI-format peak list
    
    s = info.reshape((-1, info.shape[-1]))
    c = s[:,:6]
    p = s[:,6:6+3*opts.max_peaks+1]
    r = s[:,6+3*opts.max_peaks+1:]
    
    sdat = pd.DataFrame({
            'com_x': c[:,0],
            'com_y': c[:,1],
            'lor_pk': c[:,2],
            'lor_x': c[:,3],
            'lor_y': c[:,4],
            'lor_hwhm': c[:,5],
            'center_x': r[:,0],
            'center_y': r[:,1],
            'center_refine_score': r[:,2],
            'num_peaks': p[:,-1].astype(int)})
    
    peakinfo = {
           'peakXPosRaw': p[:,:opts.max_peaks],
           'peakYPosRaw': p[:,opts.max_peaks:2*opts.max_peaks],
           'peakTotalIntensity': p[:,2*opts.max_peaks:3*opts.max_peaks],
            'nPeaks': p[:,-1].astype(int)}
    
    if shots is not None:
        sdat.index = shots.index
        sdat = pd.concat((shots, sdat), axis=1)
    return sdat, peakinfo


# %%
# Test runs. Use those to optimize the peak finding!
get_ipython().run_line_magic('matplotlib', 'ipympl')

testshot = slice(0,1000,200)
opts.load()

if testshot is not None:
    testimg = dsagg.raw_counts[testshot,...].compute()
    testdat = get_pattern_info(testimg, opts, reference, pxmask)
    out_len = testdat.shape[-1]
    testdat, peakinfo = unravel_pattern_info(testdat, opts, dsagg.shots.iloc[testshot,:])
else:
    out_len = 2+4+3*pxmask+1+2+1

plt.close('all')
for ii, shot in testdat.reset_index().iterrows():
    plt.figure(figsize=(5,5))
    pks = {k: v[ii,...] for k, v in peakinfo.items()}
    plt.imshow(testimg[ii,...],vmax=np.percentile(pks['peakTotalIntensity'],98)/5)
    plt.scatter(pks['peakXPosRaw'][:shot.num_peaks],
                pks['peakYPosRaw'][:shot.num_peaks],color='none',edgecolor='y')
    plt.scatter(shot.lor_x, shot.lor_y, color='r', marker='x')
    plt.scatter(shot.com_x, shot.com_y, color='g', marker='x')
    plt.scatter(shot.center_x, shot.center_y, color='b', marker='x')
    plt.xlim(shot.center_x-256, shot.center_x+255)
    plt.title(f'{shot.num_peaks} Peaks, {shot.center_x-shot.lor_x:.2f}, {shot.center_y-shot.lor_y:.2f} px refine offset')


# %%
# alternative version... make futures.
ftr = client.compute(compute.map_reduction_func(dsagg.raw_counts, 
                                    get_pattern_info, opts=opts,
                                    reference=reference, pxmask=pxmask,
                                    output_len=out_len,dtype=np.float), 
                     traverse=False)


# %%
alldat = ftr.result()


# %%
get_ipython().run_cell_magic('time', '', '# Now, do the final calculation\n# INSTEAD, CONSIDER APPLYING THIS TO THE SUB-DSAGGs SEPARATELY!\nalldat = compute.map_reduction_func(dsagg.raw_counts, \n                                    get_pattern_info, opts=opts,\n                                    reference=reference, pxmask=pxmask,\n                                    output_len=out_len,dtype=np.float).compute(scheduler=client.get)')


# %%
shots, peakinfo = unravel_pattern_info(alldat, opts, dsagg.shots) 


# %%
shots


# %%
# and mangle the results into nice "fake" files... later.


# %%
assert (dsagg.shots[['Event','file']] == shots[['Event','file']]).all().all()
dsagg.shots = shots


# %%
dsagg.stacks


# %%
shots.num_peaks.hist(bins=100)


# %%
# it actually seems that the COMs give better results than Lorentz!
plt.figure()
(shots.center_y-shots.com_y).hist(bins=100)


# %%
# fake-zero set
dsagg.add_stack('dummy', da.zeros_like(dsagg.raw_counts))


# %%
for k, v in peakinfo.items():
    dsagg.add_stack(k, v)


# %%
get_ipython().run_line_magic('mkdir', 'proc_data_multiproc')


# %%
get_ipython().run_line_magic('rm', '-f proc_data_multiproc/*')


# %%
dsagg.init_files(overwrite=True)


# %%
dsagg.store_tables()


# %%
get_ipython().run_line_magic('ls', '-lh proc_data_multiproc/')


# %%
dsagg.close_stacks()


# %%
dsagg.open_stacks()


# %%
# this is horribly inefficient. Instead, just write the arrays directly to hdf5
# with file-based parallelization (maybe).
# Also, already add fake-peak-positions here.
with dask.config.set(scheduler='threads'):
    dsagg.store_stacks(list(peakinfo.keys()) + ['dummy'], overwrite=True, compression='gzip')


# %%
plt.figure()
plt.imshow(testimg[ii,...],vmax=pks['peakTotalIntensity'].max()/10)


# %%
get_ipython().run_cell_magic('time', '', 'imgdat = da.compute(alldat, scheduler=client.get)')


# %%
get_ipython().run_cell_magic('time', '', 'center, peaks = da.compute(ctr_data, peak_data, scheduler=client.get)')


# %%
process_img(dsagg.raw_counts[:10].compute(), imread(opts.reference), imread(opts.pxmask))


# %%
# ctr_data = compute.map_reduction_func(stack, get_center, output_len=6)
# com = ctr_data[:,:2]
# lorentz = ctr_data[:,2:]
# ctr = lorentz[:,1:3]
# gets the peaks into CXI format
# pk_x = peak_data[:,:pkmax]
# pk_y = peak_data[:,pkmax:2*pkmax]
# pk_I = peak_data[:,2*pkmax:3*pkmax]
# npk = peak_data[:,-1].astype(int)

# radial_prof = compute.map_reduction_func(stack, proc2d.radial_proj, 
#                                          ctr[:,0], ctr[:,1], min_size=600, max_size=600,
#                                          filter_len=5, output_len=600, dtype=np.float)


# %%
refine_center(stack[:3,...].compute(), res[1][:3,...], res[4][:3,...], 
              res[5][:3,...], res[6][:3,...], res[7][:3].astype(np.int))


# %%
px.swapaxes


# %%
np.hstack([np.array([876.04466007, 357.55922995]), 0.004369143571392328])


# %%
np.stack((px, py), -1)


# %%
res[7].astype(int)


# %%



# %%
ii = 10
img = stack[ii,...].compute()
l = res[7][ii].astype(int)
px = res[4][ii, :l]
py = res[5][ii, :l]
pI = res[6][ii, :l]
plt.figure(figsize=(15,10))
plt.imshow(img,vmax=50,cmap='gray')
plt.scatter(px, py,alpha=0.2)


# %%
img[ii,:]


# %%
get_ipython().run_line_magic('pinfo2', 'pf.find_peaks')


# %%
import cython


# %%
stack[10,...]


# %%
visualize([prof, rprof, cprof])


# %%



# %%
# virtual ADF detectors NEEDS NEW FEATURES FOR CENTERING!!! 
adf1 = proc2d.apply_virtual_detector(stack, 50, 100)
adf2 = proc2d.apply_virtual_detector(stack, 100, 200)


# %%
# all the new stacks
alldata = {'center_of_mass': com, 'lorentz_fit': lorentz, 'beam_center': ctr, 'data': stack, 'adf1': adf1, 'adf2': adf2}


# %%
for lbl, stk in alldata.items():
    dsagg.add_stack(lbl, stk, overwrite=True)
dsagg.delete_stack('raw_counts', from_files=False)


# %%
limit = (dsagg.shots.file == dsagg.shots.file.unique()[0]).sum()


# %%
# on macbook, this parallelizes out of the box really well (but no IO bottleneck)
get_ipython().run_line_magic('rm', 'all_proc_dat*.h5')
get_ipython().run_line_magic('time', "da.to_hdf5('all_proc_dat.h5', alldata, compression=32004)")


# %%
# on macbook, this parallelizes out of the box really well (but no IO bottleneck)
get_ipython().run_line_magic('rm', 'all_proc_dat*.h5')
get_ipython().run_line_magic('time', "da.to_hdf5('all_proc_dat.h5', alldata)")


# %%
get_ipython().run_cell_magic('time', '', "da.to_hdf5('all_proc_dat1.h5', {k: v[:limit,...] for k, v in alldata.items()})\nda.to_hdf5('all_proc_dat2.h5', {k: v[limit:,...] for k, v in alldata.items()})")


# %%
dsagg.open_stacks()


# %%
dsagg.stacks


# %%
# this is _slower_ than to_hdf5. Strange. To check: does this get worse with larger sets, and why?
get_ipython().run_line_magic('time', 'dsagg.store_stacks(overwrite=True, compression=None, progress_bar=False)')


# %%
get_ipython().run_line_magic('pinfo2', 'dsagg.store_stacks')


# %%
get_ipython().run_cell_magic('time', '', '# Make and save the files and do the crunching\ndsagg.init_files(overwrite=True, keep_features=False);\ndsagg.store_tables(shots=True, features=True)')

# %% [markdown]
# ### Dataset initialization
# pre_process.from_raw does the following:
# * apply flat-field, saturation, and dead-pixel correction
# * find the beam center using center of mass and fit of central peak
# * shift the images accordingly
# * write everything into new, renamed files
# 
# don't be irritated by occasional `RuntimeWarning`s from `scipy.optimize`.
# 
# The following options have an influence: reference, pxmask, correct_saturation, dead_time, shutter_time, float, cam_length, com_threshold, com_xrng, lorentz_radius, lorentz_maxshift, xsize, ysize, r_adf1, r_adf2, select_query, agg_query, aggregate, scratch_dir, proc_dir, rechunk

# %%
opts.load() # re-load option file
initial = execute(pre_process.from_raw, raw_files)


# %%
get_ipython().run_cell_magic('time', '', "dsagg = dsraw.aggregate(file_suffix='_aggraw.h5', new_folder='proc_data', how={'raw_counts': 'sum'}, query='frame >=1 and frame <=5')")

# %% [markdown]
# ## Set-up of the preprocessing pipeline
# Various steps are now performed on the dsagg.raw_counts dask array. Still, none of them are actually calculated (lazy evaluation). 
# %% [markdown]
# ### A good time to optimize the peak finder
# run this on a data sub-set, and look at the output stream: `edview.py agg_peaks.stream --internal`
# 
# Then, if required, tweak the `peak_search_params` settings in `preproc.yaml` until it looks something like:
# 
# ![Nice peak finder result](doc/found_peaks.png "A good peakfinder result")

# %%
opts.load()
# we don't want to read the peaks into this workspace, so we set parse=False.
# Don't be irritated by 'waitpid() failed.' messages, that's ok.
pre_process.find_peaks(initial, opt=opts, stream_out='agg_peaks.stream', parse=False)

# %% [markdown]
# ## Centering refinement
# ...based on matching of Friedel pairs present in a single image at low resolution (flat Ewald sphere of electrons!). Make sure that the peak finder works nicely _before_ running the refinement.
# 
# After calculating and storing the re-centered stack, the peak finding is automatically invoked, and the peaks are stored in the HDF5 files depending on how `peaks_cxi` and `peaks_nexus` are set in the options file. The former makes more sense, if you want to transfer the found peaks to other files later.

# %%
opts = pre_process.PreProcOpts('preproc.yaml')
refined = execute(pre_process.refine_center, initial)

# %% [markdown]
# ### Centering check
# ...to see if the centering worked as intended, and if everything is ready for indexing. Consider running it only for a sub-set by slicing `initial` and `refined` in the first two lines. In the first cell, you will get a histogram of the change in center along x and y. 
# 
# The second cell picks 4 random patterns, and shows the right half of them three times: (left) the pattern, (center) the pattern subtracted from itself rotated by 180 degrees before refinement, (right) the same after refinement. Bad centering will cause a "dipole" shape of mated Bragg peaks. You should see, that the Bragg peaks cancel each other nicely at the end (right panel). See green arrows below. It should look something like this:
# 
# ![Centering](doc/centering.png "Centering worked well")
# 
# Don't be irritated if the central beam looks messier on the refined difference image, that's ok.

# %%
ds_original = Dataset.from_list(initial[:])
ds_refined = Dataset.from_list(refined[:])

with ds_original.Stacks() as stk_original, ds_refined.Stacks() as stk_refined:
    shifts = (ds_refined.beam_center - ds_original.beam_center).compute()

f, ax = plt.subplots(1,2)
ax[0].hist(shifts[:,0], bins=np.linspace(-2,2,50))
ax[0].set_title('x center shift / px')
ax[1].hist(shifts[:,1], bins=np.linspace(-2,2,50))
ax[1].set_title('y center shift / px');
print(np.round(sum(shifts[:,0] != 0)/len(shifts)*100,1), '% of shots were refined.')


# %%
scale = 50 # change this to adjust contrast in the image

idcs = np.arange(len(shifts))[(np.abs(shifts[:,0]) > 1) | (np.abs(shifts[:,1]) > 1)]
idcs = idcs[np.random.randint(0,len(idcs),4)]
with ds_original.Stacks() as stk_original, ds_refined.Stacks() as stk_refined:
    imgs = stk_original['centered'][idcs,...].compute()
    imgs2 = stk_refined['centered'][idcs,...].compute()

get_ipython().run_line_magic('matplotlib', 'inline')
plt.close('all')

for img, img2, shift, idx in zip(imgs, imgs2, shifts[idcs,:], idcs):
    _, ax = plt.subplots(1,3, True, True,figsize=(12,12))
    ax[0].imshow(img[:,778:778+308], vmax=scale, cmap='gray')
    ax[1].imshow(((img-np.rot90(img,2)))[:,778:778+308], vmin=-scale/2, vmax=scale/2, cmap='RdBu')
    ax[2].imshow(((img2-np.rot90(img2,2)))[:,778:778+308], vmin=-scale/2, vmax=scale/2, cmap='RdBu')
    ax[0].set_title(f'Shot {idx}')
    ax[1].set_title(f'Initial centering')
    ax[2].set_title(f'Shifted by {shift[0]:.2f}, {shift[1]:.2f}')

# %% [markdown]
# ## Ready for indexing, Preparation of sets for integration
# Congratulations! Now you have a data set of diffraction patterns that have been summed over a certain number of fractionation frames (as defined in agg_query), that is intensity-corrected, well-centered, and ready for indexing in CrystFEL. See `indexing.ipynb`.
# 
# **However**, you should not want to stop pre-processing here! There is a whole lot more to do if you want to squeeze the best out of your data - note that all of it can be done in parallel to indexing:
# * Apply pre-processing to _all_ single movie frames, instead of only the aggregated ones. This can be done much faster now using the `pre_process.broadcast` function, which reads the found peaks and beam centers from the just finished aggregated data sets and applies ('broadcasts') the results to each single movie frame.
# * Apply background subtraction using `pre_process.subtract_bg`, to either the aggregated frames, or all frames (see step above). It does not help for indexing (unless you include some sort of post-refinement), but the extracted intensities will be significantly more accurate, as CrystFELs BG subtraction does not work particularly well for electron data.
# * Once you have a set of _all_ movie frames, you can make one comprising _cumulative_ frames using `pre_process.cumulate`. In such a set, each stored pattern is a cumulative sum of patterns in that movie, i.e., corresponding to a different exposure time. These are the patterns you will want to base your final peak integration on, finding out which of the exposure times gives the best final result.
# %% [markdown]
# ### Apply pre-processing to single frames
# ...using the knowledge gained from the aggregated ones. Important options are: select_query, single_suffix, idfields, broadcast_peaks

# %%
opts.load()
allshots = execute(pre_process.broadcast, refined)

# %% [markdown]
# ### Background subtraction
# ...applied to each single frame. Alternatively, you can also apply it to the aggregated set (created above) or the cumulative set (created below), but at this point it makes most sense (I think).
# 
# The way it works is:
# - Find Bragg peaks (if `rerun_peak_finder: true`), or read them from the CXI fields in the files (if `rerun_peak_finder: false`), and label regions around them as invalid pixels. The region radius is set by `peak_radius` in the options.
# - Create a radial profile by azimuthal averaging, excluding the peak regions, and apply a median filter to it with length `filter_len` (must be odd!).
# - Calculate a background image from the radial profile, and subtract it from the original.
# 
# Unfortunately, this is still pretty slow, as the radial profile calculation is inefficient. To be optimized.

# %%
opts.load()
allshots_nobg = execute(pre_process.subtract_bg, allshots)

# %% [markdown]
# ### Cumulation
# Finally, datasets are created which contain 'cumulative' frames, that is, each frame holds the sum of all frames up to that point. An important setting is `cum_first_frame`, which allows to set the first frame from which cumulation starts, everything before is left as it is.
# 
# Say, you have an acquisition with 5 movie frames, then what you will get after `pre_process.cumulate` is:
# 
# | No. in stack       | 0 | 1   | 2     | 3     | 4     |
# |--------------------|---|-----|-------|-------|-------|
# | initial data       | 0 | 1   | 2     | 3     | 4     |
# | cum_first_frame: 0 | 0 | 0+1 | 0...2 | 0...3 | 0...4 |
# | cum_first_frame: 1 | 0 | 1   | 1+2   | 1...3 | 1...4 |
# | cum_first_frame: 2 | 0 | 1   | 2     | 2+3   | 2...4 |
# 
# Usually, you'll want to keep `cum_first_frame: 0`, unless you have too much artifacts on the first frame.

# %%
opts.load()
cumulative = execute(pre_process.cumulate, allshots_nobg)

# %% [markdown]
# ### Done!
# If you've made it to here, you should have in your proc_data folder:
# - ..._agg.h5: aggregated diffraction patterns (1 per feature/beam position), summed over a range of frames as defined in the `agg_query` parameter
# - ..._agg_refined.h5: same, with refined beam center
# - ..._all.h5: all diffraction patterns (N per feature/beam position, where N is the number of frames)
# - ..._all_nobg.h5: same, with subtracted background
# - ..._all_nobg_cumfrom0.h5: cumulative diffraction patterns (N per feature/beam position, with increasing effective exposure time)
# 
# The structure of all files is the same, and they can be read using the diffractem.Dataset class, or CrystFEL, or anything HDF5... just open it with hdfview to get an idea.
