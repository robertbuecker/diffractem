import hdf5plugin # required to access LZ4-encoded HDF5 data sets
from diffractem import version, proc2d, pre_proc_opts, nexus
from diffractem.dataset import Dataset
from tifffile import imread
import numpy as np
import os
import dask.array as da
import h5py
from dask.distributed import Client, LocalCluster, Lock
import dask
import argparse
import subprocess
import pandas as pd
import random

def _fast_correct(*args, data_key='/%/data/corrected', 
                  shots_grp='/%/shots', 
                  peaks_grp='/%/data', **kwargs):
    
    imgs, info = proc2d.analyze_and_correct(*args, **kwargs)
    store_dat = {shots_grp + '/' + k: v for k, v in info.items() if k != 'peak_data'}
    store_dat.update({peaks_grp + '/' + k: v for k, v in info['peak_data'].items()})
    store_dat[data_key] = imgs
    
    return store_dat

def quick_proc(ds, opts, client, reference=None, pxmask=None):
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
    
    label_raw = args.data_path_old.rsplit('/', 1)[-1]
    label = args.data_path_new.rsplit('/', 1)[-1]

    stack = ds.stacks[label_raw]
#     stk_del = ds.stacks['label_raw'].to_delayed().ravel()

    # get array names and shapes by correcting a single image (the last one)
    sample_res = _fast_correct(stack[-1:,...].compute(scheduler='threading'), 
                               opts=opts,
                              data_key=ds.data_pattern + '/' + label,
                              shots_grp=ds.shots_pattern,
                              peaks_grp=ds.data_pattern)
    
#     print({k: v.dtype for k, v in sample_res.items()})
    
    # initialize file structure
    for (file, subset), grp in ds.shots.groupby(['file', 'subset']):
        with h5py.File(file) as fh:
            for pattern, data in sample_res.items():
                path = pattern.replace('%', subset)
#                 print('Initializing', file, path)
                fh.require_dataset(path, 
                                    shape=(len(grp),) + data.shape[1:], 
                                    dtype=data.dtype, 
                                    chunks=(1,) + data.shape[1:], 
                                    compression=opts.compression)
            fh[ds.data_pattern.replace('%', subset)].attrs['signal'] = label
    
    # array of integers corresponding to the chunk number
    chunk_label = np.concatenate([np.repeat(ii, cs) 
                                  for ii, cs in enumerate(stack.chunks[0])])
    
    # delay objects returning the image and info dictionary
    cmp_del = [dask.delayed(_fast_correct)(raw_chk, opts) 
               for raw_chk in ds.raw_counts.to_delayed().ravel()]
    
    # file lock objects
    locks = {fn: Lock() for fn in ds.files}

    # make delay objects for writing results to file (= maximum side effects!)
    dels = []
    for chks, (cl, sht) in zip(cmp_del, ds.shots.groupby(chunk_label)):
        assert len(sht.drop_duplicates(['file','subset'])) == 1
        ii_to = sht.shot_in_subset.values
        dels.append(dask.delayed(nexus._save_single_chunk_multi)(chks,
                                                                 file=sht.file.values[0], 
                                                                 subset=sht.subset.values[0], 
                                                                 idcs=ii_to,
                                                                 lock=locks[sht.file.values[0]]
                                                                ))

    # random.shuffle(dels) # shuffling tasks to minimize concurrent file access
    chunk_info = client.compute(dels, sync=True)
    return pd.DataFrame(chunk_info, columns=['file', 'subset', 'path', 'shot_in_subset'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Quick and dirty pre-processing for Serial Electron Diffraction data')
    parser.add_argument('filename', type=str, help='List or HDF5 file or glob pattern')
    parser.add_argument('-s', '--settings', type=str, help='Option YAML file')
    parser.add_argument('-a', '--address', type=str, help='Address of dask.distributed cluster', default='127.0.0.1:8786')
    parser.add_argument('-c', '--chunksize', type=int, help='Chunk size of raw data stack. Should be integer multiple of movie stack frames!', default=100)
    parser.add_argument('-l', '--list-file', type=str, help='Name of output list file', default='processed.lst')
    parser.add_argument('-d', '--data-path-old', type=str, help='Raw data field in HDF5 file(s)', default='/%/data/raw_counts')
    parser.add_argument('-n', '--data-path-new', type=str, help='Raw data field in HDF5 file(s)', default='/%/data/corrected')
    parser.add_argument('--no-bgcorr', help='Skip background correction', action='store_true')
    parser.add_argument('ppopt', nargs=argparse.REMAINDER, help='Preprocessing options to be overriden')

    args = parser.parse_args()
    # print(extra)
    opts = pre_proc_opts.PreProcOpts(args.settings)
    print(f'Running on diffractem:', version())
    # print(f'Running on', subprocess.check_output(opts.im_exc, ' -v'))
    print(f'Current path is:', os.getcwd())
    
    print('Connecting to cluster scheduler at', args.address)
    client = Client(address=args.address)

    client.run(os.chdir, os.getcwd())

    ds_raw = Dataset.from_files(args.filename, chunking=args.chunksize)
    print('---- Have dataset ----')
    print(ds_raw)
    
    # delete undesired stacks
    delstacks = [sn for sn in ds_raw.stacks.keys() if sn != args.data_path_old.rsplit('/', 1)[-1]]
    for sn in delstacks:
        ds_raw.delete_stack(sn)

    if opts.aggregate:
        print('---- Aggregating raw data ----')
        ds_compute = ds_raw.aggregate(query=opts.agg_query, 
                                by=['sample', 'region', 'run', 'crystal_id'], 
                                how='sum', new_folder=opts.proc_dir, 
                                file_suffix=opts.agg_file_suffix)
    else:
        ds_compute = ds_raw.get_selection(query=opts.select_query,
                                    file_suffix=opts.agg_file_suffix)
    
    print('Initializing data files...')
    ds_compute.init_files(overwrite=True)

    print('Storing meta tables...')
    ds_compute.store_tables()

    print(f'Processing diffraction data... monitor progress at {client.dashboard_link} (or forward port if remote)')
    chunk_info = quick_proc(ds_compute, opts, client)
    
    # make sure that the calculation went consistent with the data set
    for (sh, sh_grp), (ch, ch_grp) in zip(ds_compute.shots.groupby(['file', 'subset']), chunk_info.groupby(['file', 'subset'])):
        if any(sh_grp.shot_in_subset.values != np.sort(np.concatenate(ch_grp.shot_in_subset.values))):
            raise ValueError(f'Incosistency between calculated data and shot list in {sh[0]}: {sh[1]} found. Please investigate.')
    
    ds_compute.write_list(args.list_file)
    
    print(f'Computation done. Processed files are in {args.list_file}')
