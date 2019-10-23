import hdf5plugin # required to access LZ4-encoded HDF5 data sets
import matplotlib.pyplot as plt
#%matplotlib ipympl
import dask.array as da
from diffractem import io, tools, proc2d, compute, nexus, proc_peaks
from diffractem.dataset import Dataset
from diffractem.stream_parser import StreamParser
import numpy as np
from tifffile import imread, imsave
import pandas as pd
import os
from glob import glob
from time import time
import json
import subprocess
import datetime
from typing import Union, Optional
from random import randrange
from shutil import copyfile

class pre_proc_opts:
    def __init__(self, fn=None):   
        self.reference = 'Ref12_reference.tif'
        self.pxmask = 'Ref12_pxmask.tif'
        self.com_threshold = 0.9
        self.com_xrng = 800
        self.lorentz_radius = 30
        self.lorentz_maxshift = 36
        self.xsize = 1556
        self.ysize = 616
        self.r_adf1 = (50, 100)
        self.r_adf2 = (100, 200)
        self.select_query = 'frame >= 0'
        self.agg_query = 'frame >= 0 and frame <= 5'
        self.agg_file_suffix = '_agg.h5'
        self.aggregate = True
        self.scratch_dir = '/scratch/diffractem'
        self.proc_dir = 'proc_data'
        self.rechunk = None
        self.crystfel_params = {'min-res': 0, 'max-res': 400, 'local-bg-radius': 4,
                    'threshold': 10, 'min-pix-count': 3, 'min-snr': 5,
                    'peaks': 'peakfinder8', 'indexing': 'none'}
        self.crystfel_params.update({'temp-dir': self.scratch_dir})
        self.crystfel_procs = 2 # number of processes
        self.im_exc = 'indexamajig'
        self.geometry = 'calibrated.geom'
        self.peaks_cxi = True
        self.half_pixel_shift = False
        self.peaks_nexus = False
        self.friedel_refine = True
        self.min_peaks = 10
        self.peak_sigma = 2
        self.refined_file_suffix = '_ref.h5'
        self.center_stack = 'beam_center'
        self.broadcast_single = True
        self.broadcast_cumulative = True
        self.single_suffix = '_all.h5'
        self.idfields = ['file_raw', 'subset', 'sample', 'crystal_id', 'region', 'run']
        self.broadcast_peaks = True
        self.cum_file_suffix = '_cum.h5'
        self.cum_stacks = ['centered']
        self.cum_first_frame = 0

        if fn is not None:
            self.read(fn)

    def read(self, fn):
        config = json.load(open(fn,'r'))
        for k, v in config.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                print('Option',k,'in',fn,'unknown.')

    def export(self, fn):
        json.dump(self.__dict__, open(fn,'w'), skipkeys=True, indent=4)

def find_peaks(ds: Union[Dataset, list, str], opt: Optional[pre_proc_opts] = None, return_cxi=True, 
                params: Optional[dict] = None, geo_params: Optional[dict] = None, procs: Optional[int] = None, 
                exc='indexamajig', **kwargs) -> Union[dict, pd.DataFrame]:
    """Handy function to find peaks using peakfinder8
    
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
    
    Returns:
        Union[dict, pd.DataFrame] -- Dataframe with peaks if return_cxi==False, otherwise dict of CXI type peak arrays
    """

    if opt is not None:
        cfpars = opt.crystfel_params
    else:
        cfpars = {'min-res': 5, 'max-res': 600, 
        'local-bg-radius': 3, 'threshold': 8, 
        'min-pix-count': 3, 'max-pix-count': 30,
        'min-snr': 3, 'int-radius': '3,4,5'}
    cfpars.update(dict({'indexing': 'none', 'peaks': 'peakfinder8'},
                    **params, **kwargs))

    rnd_id = randrange(0, 1000000)
    gfile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.geom')
    infile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.lst')
    outfile = os.path.join('.' if opt is None else opt.scratch_dir, f'pksearch_{rnd_id}.stream')
    if isinstance(ds, Dataset):
        ds.write_list(infile)
    elif isinstance(ds, str) and ds.endswith('.lst'):
        copyfile(ds, infile)
    elif isinstance(ds, str):
        with open(infile) as fh:
            fh.write(str)
    elif isinstance(ds, list):
        with open(infile) as fh:
            fh.writelines(str)

    geom = tools.make_geometry({} if geo_params is None else geo_params, gfile)
    numprocs = os.cpu_count() if procs is None else procs
    callstr = tools.call_indexamajig(infile, gfile, outfile, im_params=cfpars, procs=procs, exc=exc)
    print(callstr)
    improc1 = subprocess.run(callstr.split(' '), capture_output=True)
    print(improc1.stderr.decode())
    stream = StreamParser(outfile)
    os.remove(gfile)
    os.remove(infile)
    os.remove(outfile)

    if isinstance(ds, Dataset):
        ds.merge_stream(stream)

    if return_cxi:
        cxi = stream.get_cxi_format(half_pixel_shift=True)
        return cxi

    else:
        return stream

def from_raw(fn, opt: pre_proc_opts):

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - from_raw] '.format(datetime.datetime.now().time(), dispfn) 
        print(idstring, *args)

    t0 = time()
    dsraw = Dataset.from_list(fn)

    reference = imread(opt.reference)
    pxmask = imread(opt.pxmask)   

    os.makedirs(opt.scratch_dir, exist_ok=True)
    dsraw.open_stacks()

    if opt.aggregate:
        dsagg = dsraw.aggregate(file_suffix=opt.agg_file_suffix, new_folder=opt.proc_dir, 
                            how={'raw_counts': 'sum'}, query=opt.agg_query)
    else:
        dsagg = dsraw.get_selection(opt.select_query, new_folder=opt.proc_dir, file_suffix=opt.single_suffix)

    log(f'{dsraw.shots.shape[0]} raw, {dsagg.shots.shape[0]} aggregated/selected.')
    
    if opt.rechunk is not None:
        dsagg.rechunk_stacks(opt.rechunk)
    
    # A re-chunking of raw_counts might be a good idea at this point...
    # dsagg.add_stack('raw_counts', da.rechunk(dsagg.raw_counts, (10,-1,-1)), overwrite=True)

    # Flat-field and dead-pixel correction
    stack_ff = proc2d.apply_flatfield(dsagg.raw_counts, reference)
    stack = proc2d.correct_dead_pixels(stack_ff, pxmask, strategy='replace', replace_val=-1, mask_gaps=True)

    # Stack in central region along x (note that the gaps are not masked this time)
    xrng = slice((opt.xsize-opt.com_xrng)//2,(opt.xsize+opt.com_xrng)//2)
    stack_ct = proc2d.correct_dead_pixels(stack_ff[:,:,xrng], pxmask[:,xrng], 
                                          strategy='replace', replace_val=-1, mask_gaps=False)

    # Define COM threshold as 0.7*highest pixel (after discarding some too high ones)
    thr = stack_ct.max(axis=1).topk(10,axis=1)[:,9].reshape((-1,1,1))*opt.com_threshold
    com = proc2d.center_of_mass2(stack_ct, threshold=thr) + [[(opt.xsize-opt.com_xrng)//2, 0]]

    # Lorentzian fit in region around the found COM
    lorentz = compute.map_reduction_func(stack, proc2d.lorentz_fast, com[:,0], com[:,1], radius=opt.lorentz_radius, 
            limit=opt.lorentz_maxshift, scale=7, threads=False, output_len=4) 
    ctr = lorentz[:,1:3]

    # calculate the centered image by shifting and padding with -1
    centered = proc2d.center_image(stack, ctr[:,0], ctr[:,1], opt.xsize, opt.ysize, -1, parallel=True)

    # add the new stacks to the aggregated dataset
    alldata = {'center_of_mass': com, 
                'lorentz_fit': lorentz, 
                'beam_center': ctr, 
                'centered': centered, 
               'pxmask_centered': (centered != -1).astype(np.uint16), 
               'adf1': proc2d.apply_virtual_detector(centered, opt.r_adf1[0], opt.r_adf1[1]), 
               'adf2': proc2d.apply_virtual_detector(centered, opt.r_adf2[0], opt.r_adf2[1])}
    for lbl, stk in alldata.items():
        dsagg.add_stack(lbl, stk, overwrite=True)

    # make the files and crunch the data
    try:
        dsagg.init_files(overwrite=True)
        dsagg.store_tables(shots=True, features=True)
        dsagg.open_stacks()
        dsagg.delete_stack('raw_counts', from_files=False) # we don't need the raw counts in the new files
        dsagg.store_stacks(overwrite=True, progress_bar=False) # this does the actual calculation

        log('Finished first centering', dsagg.centered.shape[0], 'shots after', time()-t0, 'seconds')

    except Exception as err:
        log('Raw processing of failed.')
        raise err

    finally:    
        dsagg.close_stacks()   
        dsraw.close_stacks()
     
    # write file list for crystfel
    list_name = os.path.join(opt.scratch_dir,os.path.basename(dsagg.files[0]).rsplit('.')[0]) + '_ff.lst'
    dsagg.write_list(list_name)
    stream_name = list_name.rsplit('.',1)[0] + f'_peaks.stream'

    # run CrystFEL peak finding (indexamajig with indexing=none)
    callstr = tools.call_indexamajig(list_name, opt.geometry, stream_name, 
                                     im_params=opt.crystfel_params, procs=opt.crystfel_procs, 
                                     exc=opt.im_exc)
    
    log('Running indexamajig:', callstr)
    improc1 = subprocess.run(callstr.split(' '), capture_output=True)
    #print(improc1)

    # parse stream file (but don't merge into data set yet)
    stream = StreamParser(stream_name)
    log('Indexamajig found',stream.peaks.shape[0],'peaks.')
    
    # if desired, do Friedel mate refinement
    if not opt.friedel_refine:
        dsagg.merge_stream(stream)
        dsagg.store_tables(shots=True, peaks=opt.peaks_nexus)

    else:
        # get Friedel-refined center from stream file
        ctr_fr = proc_peaks.center_friedel(stream.peaks, dsagg.shots, 
                                    p0=[opt.xsize//2, opt.ysize//2], 
                                    sigma=opt.peak_sigma, minpeaks=opt.min_peaks)

        try:
            dsagg.open_stacks()
            # re-center, based on already-centered images
            centered2 = proc2d.center_image(dsagg.centered, ctr_fr['beam_x'].values, ctr_fr['beam_y'].values, opt.xsize, opt.ysize, -1)
            ctr2 = (ctr_fr[['beam_x', 'beam_y']].values - [[opt.xsize//2, opt.ysize//2]]) + da.ceil(dsagg.beam_center)

            dsagg.add_stack('centered', centered2, overwrite=True)
            dsagg.add_stack('pxmask_centered', (centered2 != -1).astype(np.uint16), overwrite=True)
            dsagg.add_stack('beam_center', ctr2, overwrite=True)

            dsagg.change_filenames(opt.refined_file_suffix)
            dsagg.init_files(overwrite=True, keep_features=True)
            dsagg.store_tables(shots=True)
            dsagg.store_stacks(overwrite=True, progress_bar=False)
        except Exception as err:
            log('Friedel refinement raised an error:')
            raise err
        finally:        
            dsagg.close_stacks()
            del centered2

        #re-run peak finder
        improc1 = subprocess.run(callstr.split(' '), capture_output=True)
        stream = StreamParser(stream_name)
        dsagg.merge_stream(stream)
        dsagg.store_tables(shots=True, peaks=opt.peaks_nexus)

        log('Finished refined centering', dsagg.centered.shape[0], 'shots after', time()-t0, 'seconds')
    
    # export peaks to CXI-format arrays
    if opt.peaks_cxi:
        cxidata = stream.get_cxi_format(shots=dsagg.shots, half_pixel_shift=opt.half_pixel_shift)
        dsagg.open_stacks()
        for k, v in cxidata.items():
            dsagg.add_stack(k, v, overwrite=True)
        dsagg.store_stacks(list(cxidata.keys()), progress_bar=False)
        dsagg.close_stacks()

    log('Finished raw processing with', dsagg.centered.shape[0], 'shots after', time()-t0, 'seconds')
    return dsagg.files


def broadcast(fn, opt: pre_proc_opts):

    if isinstance(fn, list) and len(fn) == 1:
        fn = fn[0]

    def log(*args):
        if isinstance(fn, list):
            dispfn = os.path.basename(fn[0]) + ' etc.'
        else:
            dispfn = os.path.basename(fn)
        idstring = '[{} - {} - broadcast] '.format(datetime.datetime.now().time(), dispfn) 
        print(idstring, *args)

    t0 = time()
    dsagg = Dataset.from_list(fn, load_tables=False)
    dsagg.open_stacks()
    dsraw = Dataset.from_list(list(dsagg.shots.file_raw.unique()))
    dsraw.open_stacks()
    dssel = dsraw.get_selection(opt.select_query, file_suffix=opt.single_suffix, new_folder=opt.proc_dir)

    reference = imread(opt.reference)
    pxmask = imread(opt.pxmask)   

    log(f'{dsraw.shots.shape[0]} raw, {dssel.shots.shape[0]} selected, {dsagg.shots.shape[0]} aggregated.')

    # And now: the interesting part...
    dsagg.shots['from_id'] = range(dsagg.shots.shape[0])
    shots = dssel.shots.merge(dsagg.shots[opt.idfields + ['from_id']], on=opt.idfields, validate='m:1')
    
    # get the broadcasted image centers
    ctr = dsagg.stacks[opt.center_stack][shots.from_id.values,:]  
   
    # Flat-field and dead-pixel correction
    stack_rechunked = dssel.raw_counts.rechunk({0: ctr.chunks[0]}) # re-chunk the raw data
    stack_ff = proc2d.apply_flatfield(stack_rechunked, reference)
    stack = proc2d.correct_dead_pixels(stack_ff, pxmask, strategy='replace', replace_val=-1, mask_gaps=True)
    centered = proc2d.center_image(stack, ctr[:,0], ctr[:,1], opt.xsize, opt.ysize, -1, parallel=True)
    
    # add the new stacks to the aggregated dataset
    alldata = {'center_of_mass': dsagg.stacks['center_of_mass'][shots.from_id.values,...], 
               'lorentz_fit': dsagg.stacks['lorentz_fit'][shots.from_id.values,...], 
               'beam_center': ctr, 
               'centered': centered.astype(np.int16), 
               'pxmask_centered': (centered != -1).astype(np.uint16), 
               'adf1': proc2d.apply_virtual_detector(centered, opt.r_adf1[0], opt.r_adf1[1]), 
               'adf2': proc2d.apply_virtual_detector(centered, opt.r_adf2[0], opt.r_adf2[1])
              }
    
    if opt.broadcast_peaks:
        alldata.update({
            'nPeaks': dsagg.stacks['nPeaks'][shots.from_id.values,...],
            'peakTotalIntensity': dsagg.stacks['peakTotalIntensity'][shots.from_id.values,...],
            'peakXPosRaw': dsagg.stacks['peakXPosRaw'][shots.from_id.values,...],
            'peakYPosRaw': dsagg.stacks['peakYPosRaw'][shots.from_id.values,...],
        })
        
    for lbl, stk in alldata.items():
        dssel.add_stack(lbl, stk, overwrite=True)
    
    try:
        dssel.init_files(overwrite=True)
        dssel.store_tables(shots=True, features=True)
        dssel.open_stacks()
        dssel.delete_stack('raw_counts', from_files=False) # we don't need the raw counts in the new files
        dssel.store_stacks(overwrite=True, progress_bar=False) # this does the actual calculation
        log('Finished with', dssel.centered.shape[0], 'shots after', time()-t0, 'seconds')
 
    except Exception as err:
        log('Broadcast processing failed.')
        raise err
       
    finally:    
        dsagg.close_stacks()
        dsraw.close_stacks()
        dssel.close_stacks()
    
    return dssel.files


def cumulate(fn, opt: pre_proc_opts):
    #fn = 'proc_data/DUT-67_000_00000_all.h5'

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
    dssel.open_stacks()

    # chunks for aggregation
    chunks = tuple(dssel.shots.groupby(opt.idfields).count()['selected'].values)
    for k, stk in dssel.stacks.items():
        if stk.chunks[0] != chunks:
            if k == 'index':
                continue
            log(k,'needs rechunking...')
            dssel.add_stack(k, stk.rechunk({0: chunks}), overwrite=True)
    dssel._zchunks = chunks

    def cumfunc(movie):
        movie_out = movie
        movie_out[opt.cum_first_frame:,...] = np.cumsum(movie[opt.cum_first_frame:,...], axis=0)
        return movie_out

    for k in opt.cum_stacks:
        dssel.stacks[k] = dssel.stacks[k].map_blocks(cumfunc, dtype=dssel.stacks[k].dtype)

    dssel.change_filenames(opt.cum_file_suffix)    
    dssel.init_files(overwrite=True, keep_features=False)
    dssel.store_tables()
    
    try:
        dssel.open_stacks()        
        dssel.store_stacks(overwrite=True, progress_bar=False)

    except Exception as err:
        log('Cumulative processing failed.')
        raise err

    finally:
        dssel.close_stacks()
    
    return dssel.files
