import hdf5plugin # required to access LZ4-encoded HDF5 data sets
import dask.array as da
from diffractem import io, tools, proc2d, compute, nexus, proc_peaks
from diffractem.dataset import Dataset
from diffractem.stream_parser import StreamParser
import numpy as np
from tifffile import imread
import os
from time import time
import json
import subprocess
import datetime
from typing import Union, Optional, Callable
from random import randrange
from shutil import copyfile
from concurrent.futures import ProcessPoolExecutor, FIRST_EXCEPTION, wait
import dask
import yaml
import pprint
from diffractem.pre_proc_opts import PreProcOpts

"""
Pre-processing macros for diffractem.

They operate on files or lists thereof, according to the settings made in an PreProcOpts object.
(With the exception of find_peaks, which does not require an opts object, but settings can be made
in the options directly)

Mostly, the functions here call functions from proc2d in a more or less smart sequence.
"""

def find_peaks(ds: Union[Dataset, list, str], opt: Optional[PreProcOpts] = None,
               from_cxi=False, return_cxi=True, revalidate_cxi=False, merge_peaks=True,
               params: Optional[dict] = None, geo_params: Optional[dict] = None, procs: Optional[int] = None,
               exc='indexamajig', stream_out: Optional[str] = None, parse=True, **kwargs) \
        -> Union[dict, StreamParser]:
    """Handy function to find peaks using peakfinder8 or extract a stream/pandas list from CXI-format stored peaks
    
    Arguments:
        ds {Union[Dataset, list, str]} -- dataset, list file, single nexus file, or list of files.
            If a Dataset is supplied, the peakdata will be merged into it.
    
    Keyword Arguments:
        opt {Optional[pre_proc_opts]} -- pre-processing option object containing defaults for
            indexamajig (default: {None})
        return_cxi {bool} -- return a dict of CXI peak arrays instead of pandas dataframe (default: {True})
        params {Optional[dict]} -- additional parameter for indexamajig (default: {None})
        geo_params {Optional[dict]} -- additional parameters for geometry file, e.g. clen (default: {None})
        procs {Optional[int]} -- number of processes. If None, uses all available cores (default: {None})
        exc {str} -- Path of indexamajig executable (default: {'indexamajig'})
        stream_out {str} -- File name of stream file, if it should be retained.
    
    Returns:
        Union[dict, pd.DataFrame] -- Dataframe with peaks if return_cxi==False, otherwise dict of CXI type peak arrays
    """

    #print(type(ds))
    #print(isinstance(ds, type(ds)))

    if opt is not None:
        cfpars = opt.peak_search_params
        exc = opt.im_exc
    else:
        cfpars = {'min-res': 5, 'max-res': 600,
                  'local-bg-radius': 3, 'threshold': 8,
                  'min-pix-count': 3,
                  'min-snr': 3, 'int-radius': '3,4,5', 'peaks': 'peakfinder8'}

    params = {} if params is None else params
    
    if from_cxi:
        cfpars.update(
            dict({'indexing': 'none', 'peaks': 'cxi', 'hdf5-peaks': '/%/data', 'no-revalidate': not revalidate_cxi},
                 **params, **kwargs))
    else:
        cfpars.update(dict({'indexing': 'none'},
                    **params, **kwargs))

    rnd_id = randrange(0, 1000000)
    gfile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.geom')
    infile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.lst')

    if (not parse) and (stream_out is None):
        raise ValueError('You must either parse the peaks, or give an output name for the stream file.')

    if stream_out is None:
        outfile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.stream')
    else:
        outfile = stream_out
    if isinstance(ds, Dataset):
        ds.write_list(infile)
        # print('Wrote', infile)
    elif isinstance(ds, str) and ds.endswith('.lst'):
        copyfile(ds, infile)
    elif isinstance(ds, str):
        with open(infile, 'w') as fh:
            fh.write(ds)
    elif isinstance(ds, list):
        with open(infile, 'w') as fh:
            fh.writelines([l + '\n' for l in ds])
    else:
        raise ValueError('ds must be a list, string or Dataset.')

    callstr = tools.call_indexamajig(infile, gfile, outfile, im_params=cfpars, procs=procs, exc=exc)
    #print(callstr)
    improc1 = subprocess.run(callstr.split(' '), capture_output=True)
    if opt is None or opt.verbose:
        print(improc1.stderr.decode())
    else:
        print('\n'.join([l for l in improc1.stderr.decode().split('\n') if l.startswith('Final') or l.lower().startswith('warning')]))
    os.remove(gfile)
    os.remove(infile)

    if parse and (stream_out is None):
        stream = StreamParser(outfile)
        os.remove(outfile)
    elif parse and (stream_out is not None):
        stream = StreamParser(outfile)
    elif not parse and (stream_out is not None):
        return outfile
    else:
        raise RuntimeError('This should not happen.')

    if isinstance(ds, Dataset) and merge_peaks:
        ds.merge_stream(stream)

    if return_cxi:
        cxi = stream.get_cxi_format(half_pixel_shift=True)
        return cxi

    else:
        return stream

def from_raw(fn, opt: PreProcOpts):

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if not (opt.verbose or any([isinstance(err, Exception) for e in args])):
            return        
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - from_raw] '.format(datetime.datetime.now().time(), dispfn) 
        print(idstring, *args)

    t0 = time()
    dsraw = Dataset().from_list(fn)

    reference = imread(opt.reference)
    pxmask = imread(opt.pxmask)   

    os.makedirs(opt.scratch_dir, exist_ok=True)
    os.makedirs(opt.proc_dir, exist_ok=True)
    dsraw.open_stacks(readonly=True)

    if opt.aggregate:
        dsagg = dsraw.aggregate(file_suffix=opt.agg_file_suffix, new_folder=opt.proc_dir, force_commensurate=False,
                            how={'raw_counts': 'sum'}, query=opt.agg_query)
    else:
        dsagg = dsraw.get_selection(opt.agg_query, new_folder=opt.proc_dir, file_suffix=opt.agg_file_suffix)

    log(f'{dsraw.shots.shape[0]} raw, {dsagg.shots.shape[0]} aggregated/selected.')
    
    if opt.rechunk is not None:
        dsagg.rechunk_stacks(opt.rechunk)

    # Saturation, flat-field and dead-pixel correction
    if opt.correct_saturation:
        stack_ff = proc2d.apply_flatfield(proc2d.apply_saturation_correction(
            dsagg.raw_counts, opt.shutter_time, opt.dead_time), reference)
    else:
        stack_ff = proc2d.apply_flatfield(dsagg.raw_counts, reference)

    stack = proc2d.correct_dead_pixels(stack_ff, pxmask, strategy='replace', replace_val=-1, mask_gaps=True)

    # Stack in central region along x (note that the gaps are not masked this time)
    xrng = slice((opt.xsize - opt.com_xrng) // 2, (opt.xsize + opt.com_xrng) // 2)
    stack_ct = proc2d.correct_dead_pixels(stack_ff[:, :, xrng], pxmask[:, xrng],
                                          strategy='replace', replace_val=-1, mask_gaps=False)

    # Define COM threshold as fraction of highest pixel (after discarding some too high ones)
    thr = stack_ct.max(axis=1).topk(10, axis=1)[:, 9].reshape((-1, 1, 1)) * opt.com_threshold
    com = proc2d.center_of_mass2(stack_ct, threshold=thr) + [[(opt.xsize - opt.com_xrng) // 2, 0]]

    # Lorentzian fit in region around the found COM
    lorentz = compute.map_reduction_func(stack, proc2d.lorentz_fast, com[:, 0], com[:, 1], radius=opt.lorentz_radius,
                                         limit=opt.lorentz_maxshift, scale=7, threads=False, output_len=4)
    ctr = lorentz[:, 1:3]

    # calculate the centered image by shifting and padding with -1
    centered = proc2d.center_image(stack, ctr[:, 0], ctr[:, 1], opt.xsize, opt.ysize, -1, parallel=True)

    # add the new stacks to the aggregated dataset
    alldata = {'center_of_mass': com,
               'lorentz_fit': lorentz,
               'beam_center': ctr,
               'centered': centered,
               'pxmask_centered': (centered != -1).astype(np.uint16),
               'adf1': proc2d.apply_virtual_detector(centered, opt.r_adf1[0], opt.r_adf1[1]),
               'adf2': proc2d.apply_virtual_detector(centered, opt.r_adf2[0], opt.r_adf2[1])}
    for lbl, stk in alldata.items():
        print('adding', lbl, stk.shape)
        dsagg.add_stack(lbl, stk, overwrite=True)

    # make the files and crunch the data
    try:
        dsagg.init_files(overwrite=True)
        dsagg.store_tables(shots=True, features=True)
        dsagg.open_stacks(readonly=False)
        dsagg.delete_stack('raw_counts', from_files=False) # we don't need the raw counts in the new files
        dsagg.store_stacks(overwrite=True, progress_bar=False) # this does the actual calculation

        log('Finished first centering', dsagg.centered.shape[0], 'shots after', time()-t0, 'seconds')

    except Exception as err:
        log('Raw processing failed.', err)
        raise err

    finally:    
        dsagg.close_stacks()   
        dsraw.close_stacks()

    return dsagg.files


def refine_center(fn, opt: PreProcOpts):
    """Refines the centering of diffraction patterns based on Friedel mate positions.
    
    Arguments:
        fn {str} -- [file/list name of input files, can contain wildcards]
        opt {PreProcOpts} -- [pre-processing options]
    
    Raises:
        err: [description]
    
    Returns:
        [list] -- [output files]
    """

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if not (opt.verbose or any([isinstance(err, Exception) for e in args])):
            return
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - refine_center] '.format(datetime.datetime.now().time(), dispfn) 
        print(idstring, *args)

    ds = Dataset.from_list(fn)
    stream = find_peaks(ds, opt=opt, merge_peaks=False, 
        return_cxi=False, geo_params={'clen': opt.cam_length}, exc=opt.im_exc)

    p0 = [opt.xsize//2, opt.ysize//2]

    # get Friedel-refined center from stream file
    ctr = proc_peaks.center_friedel(stream.peaks, ds.shots,
                                    p0=p0,
                                    sigma=opt.peak_sigma, minpeaks=opt.min_peaks,
                                    maxres=opt.friedel_max_radius)

    maxcorr = ctr['friedel_cost'].values
    changed = np.logical_not(np.isnan(maxcorr))

    with ds.Stacks() as stk:
        beam_center_old = stk['beam_center'].compute()  # previous beam center

    beam_center_new = beam_center_old.copy()
    beam_center_new[changed, :] = np.ceil(beam_center_old[changed, :]) + ctr.loc[
        changed, ['beam_x', 'beam_y']].values - p0
    if (np.abs(np.mean(beam_center_new - beam_center_old, axis=0) > .5)).any():
        log('WARNING: average shift is larger than 0.5!')

    # visualization
    log('{:g}% of shots refined. \n'.format((1 - np.isnan(maxcorr).sum() / len(maxcorr)) * 100),
        'Shift standard deviation: {} \n'.format(np.std(beam_center_new - beam_center_old, axis=0)),
        'Average shift: {} \n'.format(np.mean(beam_center_new - beam_center_old, axis=0)))

    # make new files and add the shifted images
    try:
        ds.open_stacks(readonly=False)
        centered2 = proc2d.center_image(ds.centered, ctr['beam_x'].values, ctr['beam_y'].values, 1556, 616, -1)
        ds.add_stack('centered', centered2, overwrite=True)
        ds.add_stack('pxmask_centered', (centered2 != -1).astype(np.uint16), overwrite=True)
        ds.add_stack('beam_center', beam_center_new, overwrite=True)
        ds.change_filenames(opt.refined_file_suffix)
        print(ds.files)
        ds.init_files(keep_features=False, overwrite=True)
        ds.store_tables(shots=True, features=True)
        ds.open_stacks(readonly=False)
        ds.store_stacks(overwrite=True, progress_bar=False)
        ds.close_stacks()
        del centered2
    except Exception as err:
        log('Error during storing center-refined images', err)
        raise err
    finally:
        ds.close_stacks()

    # run peak finder again, this time on the refined images
    pks_cxi = find_peaks(ds, opt=opt, merge_peaks=opt.peaks_nexus,
                         return_cxi=True, geo_params={'clen': opt.cam_length}, exc=opt.im_exc)

    # export peaks to CXI-format arrays
    if opt.peaks_cxi:
        with ds.Stacks() as stk:
            for k, v in pks_cxi.items():
                if k in stk:
                    ds.delete_stack(k, from_files=True)
                ds.add_stack(k, v, overwrite=True)
            ds.store_stacks(list(pks_cxi.keys()), progress_bar=True, overwrite=True)

    return ds.files


def subtract_bg(fn, opt: PreProcOpts):
    """Subtracts the background of a diffraction pattern by azimuthal integration excluding the Bragg peaks.
    
    Arguments:
        fn {function} -- [description]
        opt {PreProcOpts} -- [description]
    
    Returns:
        [type] -- [description]
    """

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if not (opt.verbose or any([isinstance(err, Exception) for e in args])):
            return
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - subtract_bg] '.format(datetime.datetime.now().time(), dispfn) 
        print(idstring, *args)

    ds = Dataset().from_list(fn)
    ds.open_stacks(readonly=False)

    if opt.rerun_peak_finder:
        pks = find_peaks(ds, opt=opt)
        nPeaks = da.from_array(pks['nPeaks'][:, np.newaxis, np.newaxis], chunks=(ds.centered.chunks[0], 1, 1))
        peakX = da.from_array(pks['peakXPosRaw'][:, :, np.newaxis], chunks=(ds.centered.chunks[0], -1, 1))
        peakY = da.from_array(pks['peakYPosRaw'][:, :, np.newaxis], chunks=(ds.centered.chunks[0], -1, 1))
    else:
        nPeaks = ds.nPeaks[:, np.newaxis, np.newaxis]
        peakX = ds.peakXPosRaw[:, :, np.newaxis]
        peakY = ds.peakYPosRaw[:, :, np.newaxis]

    original = ds.centered
    bg_corrected = da.map_blocks(proc2d.remove_background, original, original.shape[2] / 2, original.shape[1] / 2,
                                 nPeaks, peakX, peakY, peak_radius=opt.peak_radius, filter_len=opt.filter_len,
                                 dtype=np.float32 if opt.float else np.int32, chunks=original.chunks)

    ds.add_stack('centered', bg_corrected, overwrite=True)
    ds.change_filenames(opt.nobg_file_suffix)
    ds.init_files(keep_features=False, overwrite=True)
    ds.store_tables(shots=True, features=True, peaks=False)
    ds.open_stacks(readonly=False)

    # for lbl in ['nPeaks', 'peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw']:
    #    if lbl in ds.stacks:
    #        ds.delete_stack(lbl, from_files=False)

    try:
        ds.store_stacks(overwrite=True, progress_bar=False)
    except Exception as err:
        log('Error during background correction:', err)
        raise err
    finally:
        ds.close_stacks()

    return ds.files


def broadcast(fn, opt: PreProcOpts):
    """Pre-processes in one go a dataset comprising movie frames, by transferring the found beam center
    positions and diffraction spots (in CXI format) from an aggregated set processed earlier.
    
    Arguments:
        fn {function} -- [description]
        opt {PreProcOpts} -- [description]
    
    Raises:
        err: [description]
    
    Returns:
        [type] -- [description]
    """

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if not (opt.verbose or any([isinstance(err, Exception) for e in args])):
            return
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - broadcast] '.format(datetime.datetime.now().time(), dispfn)
        print(idstring, *args)

    t0 = time()
    dsagg = Dataset.from_list(fn, load_tables=False)
    dsraw = Dataset.from_list(list(dsagg.shots.file_raw.unique()))
    dsraw.shots['file_raw'] = dsraw.shots['file'] # required for association later

    reference = imread(opt.reference)
    pxmask = imread(opt.pxmask)

    # And now: the interesting part...
    dsagg.shots['from_id'] = range(dsagg.shots.shape[0])    # label original (aggregated shots)

    in_agg = dsraw.shots[opt.idfields].merge(dsagg.shots[opt.idfields + ['selected']], on=opt.idfields, how='left')['selected'].fillna(False)
    dsraw.shots['selected'] = in_agg

    try:

        dsraw.open_stacks(readonly=True)
        dssel = dsraw.get_selection(f'({opt.select_query}) and selected', file_suffix=opt.single_suffix, new_folder=opt.proc_dir)
        shots = dssel.shots.merge(dsagg.shots[opt.idfields + ['from_id']], on=opt.idfields, validate='m:1') # _inner_ merge
        
        log(f'{dsraw.shots.shape[0]} raw, {dssel.shots.shape[0]} selected, {dsagg.shots.shape[0]} aggregated.')

        # get the broadcasted image centers
        dsagg.open_stacks(readonly=True)
        ctr = dsagg.stacks[opt.center_stack][shots.from_id.values, :]

        # Flat-field and dead-pixel correction
        stack_rechunked = dssel.raw_counts.rechunk({0: ctr.chunks[0]})  # select and re-chunk the raw data
        if opt.correct_saturation:
            stack_ff = proc2d.apply_flatfield(proc2d.apply_saturation_correction(
                stack_rechunked, opt.shutter_time, opt.dead_time), reference)
        else:
            stack_ff = proc2d.apply_flatfield(stack_rechunked, reference)
        stack = proc2d.correct_dead_pixels(stack_ff, pxmask, strategy='replace', replace_val=-1, mask_gaps=True)
        centered = proc2d.center_image(stack, ctr[:, 0], ctr[:, 1], opt.xsize, opt.ysize, -1, parallel=True)

        # add the new stacks to the aggregated dataset
        alldata = {'center_of_mass': dsagg.stacks['center_of_mass'][shots.from_id.values, ...],
                'lorentz_fit': dsagg.stacks['lorentz_fit'][shots.from_id.values, ...],
                'beam_center': ctr,
                'centered': centered.astype(np.float32) if opt.float else centered.astype(np.int16),
                'pxmask_centered': (centered != -1).astype(np.uint16),
                'adf1': proc2d.apply_virtual_detector(centered, opt.r_adf1[0], opt.r_adf1[1]),
                'adf2': proc2d.apply_virtual_detector(centered, opt.r_adf2[0], opt.r_adf2[1])
                }

        if opt.broadcast_peaks:
            alldata.update({
                'nPeaks': dsagg.stacks['nPeaks'][shots.from_id.values, ...],
                'peakTotalIntensity': dsagg.stacks['peakTotalIntensity'][shots.from_id.values, ...],
                'peakXPosRaw': dsagg.stacks['peakXPosRaw'][shots.from_id.values, ...],
                'peakYPosRaw': dsagg.stacks['peakYPosRaw'][shots.from_id.values, ...],
            })
            
        for lbl, stk in alldata.items():
            dssel.add_stack(lbl, stk, overwrite=True)

        dssel.init_files(overwrite=True)
        dssel.store_tables(shots=True, features=True)
        dssel.open_stacks(readonly=False)
        dssel.delete_stack('raw_counts', from_files=False) # we don't need the raw counts in the new files
        dssel.store_stacks(overwrite=True, progress_bar=False) # this does the actual calculation
        log('Finished with', dssel.centered.shape[0], 'shots after', time()-t0, 'seconds')
 
    except Exception as err:
        log('Broadcast processing failed:', err)
        raise err
       
    finally:    
        dsagg.close_stacks()
        dsraw.close_stacks()
        dssel.close_stacks()
    
    return dssel.files


def cumulate(fn, opt: PreProcOpts):
    """Applies cumulative summation to a data set comprising movie frame stacks. At the moment, requires
    the summed frame stacks to have the same shape as the raw data.
    
    Arguments:
        fn {function} -- [description]
        opt {PreProcOpts} -- [description]
    
    Raises:
        err: [description]
    
    Returns:
        [type] -- [description]
    """

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - cumulate] '.format(datetime.datetime.now().time(), dispfn)
        print(idstring, *args)

    dssel = Dataset().from_list(fn)
    log('Cumulating from frame', opt.cum_first_frame)
    dssel.open_stacks(readonly=False)

    # chunks for aggregation
    chunks = tuple(dssel.shots.groupby(opt.idfields).count()['selected'].values)
    for k, stk in dssel.stacks.items():
        if stk.chunks[0] != chunks:
            if k == 'index':
                continue
            log(k, 'needs rechunking...')
            dssel.add_stack(k, stk.rechunk({0: chunks}), overwrite=True)
    dssel._zchunks = chunks

    def cumfunc(movie):
        movie_out = movie
        movie_out[opt.cum_first_frame:, ...] = np.cumsum(movie[opt.cum_first_frame:, ...], axis=0)
        return movie_out

    for k in opt.cum_stacks:
        dssel.stacks[k] = dssel.stacks[k].map_blocks(cumfunc, dtype=dssel.stacks[k].dtype)

    dssel.change_filenames(opt.cum_file_suffix)
    dssel.init_files(overwrite=True, keep_features=False)
    log('File initialized, writing tables...')
    dssel.store_tables(shots=True, features=True, peaks=False)

    try:
        dssel.open_stacks(readonly=False)
        log('Writing stack data...')
        dssel.store_stacks(overwrite=True, progress_bar=False)

    except Exception as err:
        log('Cumulative processing failed.')
        raise err

    finally:
        dssel.close_stacks()
        log('Cumulation done.')

    return dssel.files
