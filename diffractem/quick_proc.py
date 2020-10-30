import hdf5plugin # required to access LZ4-encoded HDF5 data sets
from diffractem import version, proc2d, pre_proc_opts, nexus, io
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
from warnings import warn
from time import sleep

def _fast_correct(*args, data_key='/%/data/corrected', 
                  shots_grp='/%/shots', 
                  peaks_grp='/%/data', **kwargs):
    
    imgs, info = proc2d.analyze_and_correct(*args, **kwargs)
    store_dat = {shots_grp + '/' + k: v for k, v in info.items() if k != 'peak_data'}
    store_dat.update({peaks_grp + '/' + k: v for k, v in info['peak_data'].items()})
    store_dat[data_key] = imgs
    
    return store_dat

def quick_proc(ds, opts, label_raw, label, client, reference=None, pxmask=None):
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask

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
        with h5py.File(file, 'a') as fh:
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

def main():

    parser = argparse.ArgumentParser(description='Quick and dirty pre-processing for Serial Electron Diffraction data', 
                                     allow_abbrev=False, epilog='Any other options are passed on as modification to the option file')
    parser.add_argument('filename', type=str, nargs='*', help='List or HDF5 file or glob pattern. Glob pattern must be given in SINGLE quotes.')
    parser.add_argument('-s', '--settings', type=str, help='Option YAML file. Defaults to \'preproc.yaml\'.', default='preproc.yaml')
    parser.add_argument('-A', '--address', type=str, help='Address of an existing dask.distributed cluster to use instead of making a new one. Defaults to making a new one.', default=None)
    parser.add_argument('-N', '--nprocs', type=int, help='Number of processes of a new dask.distributed cluster. Defaults to letting dask decide.', default=None)
    parser.add_argument('-L', '--local-directory', type=str, help='Fast (scratch) directory for computations. Defaults to the current directory.', default=None)
    parser.add_argument('-c', '--chunksize', type=int, help='Chunk size of raw data stack. Should be integer multiple of movie stack frames! Defaults to 100.', default=100)
    parser.add_argument('-l', '--list-file', type=str, help='Name of output list file', default='processed.lst')
    parser.add_argument('-w', '--wait-for-files', help='Wait for files matching wildcard pattern', action='store_true')
    parser.add_argument('--include-existing', help='When using -w/--wait-for-file, also include existing files', action='store_true')
    parser.add_argument('--append', help='Append to list instead of overwrite', action='store_true')
    parser.add_argument('-d', '--data-path-old', type=str, help='Raw data field in HDF5 file(s). Defaults to /entry/data/raw_data', default='/%/data/raw_counts')
    parser.add_argument('-n', '--data-path-new', type=str, help='Corrected data field in HDF5 file(s). Defaults to /entry/data/corrected', default='/%/data/corrected')
    parser.add_argument('--no-bgcorr', help='Skip background correction', action='store_true')
    parser.add_argument('--no-validate', help='Do not validate files before attempting to process', action='store_true')
    # parser.add_argument('ppopt', nargs=argparse.REMAINDER, help='Preprocessing options to be overriden')

    args, extra = parser.parse_known_args()
    # print(args, extra)
    # raise RuntimeError('thus far!')
    opts = pre_proc_opts.PreProcOpts(args.settings)
     
    label_raw = args.data_path_old.rsplit('/', 1)[-1]
    label = args.data_path_new.rsplit('/', 1)[-1]
       
    if extra:
        # If extra arguments have been supplied, overwrite existing values
        opt_parser = argparse.ArgumentParser()
        for k, v in opts.__dict__.items():
            opt_parser.add_argument('--' + k, type=type(v), default=None)
        opts2 = opt_parser.parse_args(extra)
        
        for k, v in vars(opts2).items():
            if v is not None:
                if type(v) != type(opts.__dict__[k]):
                    warn('Mismatch of data types in overriden argument!', RuntimeWarning)
                print(f'Overriding option file setting {k} = {opts.__dict__[k]} ({type(opts.__dict__[k])}). ',
                    f'New value is {v} ({type(v)})')
                opts.__dict__[k] = v
    
    # raise RuntimeError('thus far!')
    print(f'Running on diffractem:', version())
    print(f'Current path is:', os.getcwd())
    
    # client = Client()
    if args.address is not None:
        print('Connecting to cluster scheduler at', args.address)
    
        try:
            client = Client(address=args.address, timeout=3)
        except:
            print(f'\n----\nThere seems to be no dask.distributed scheduler running at {args.address}.\n'
                f'Please double-check or start one by either omitting the --address option.')
            return
    else:
        print('Creating a dask.distributed cluster...')
        client = Client(n_workers=args.nprocs, local_directory=args.local_directory, processes=True)   
        print('\n\n---\nStarted dask.distributed cluster:')
        print(client)
        print('You can access the dashboard for monitoring at: ', client.dashboard_link)
        
    
    client.run(os.chdir, os.getcwd())
    
    if len(args.filename) == 1:
        args.filename = args.filename[0]
    
    # print(args.filename)
    seen_raw_files = [] if args.include_existing else io.expand_files(args.filename)

    while True:
    
        if args.wait_for_files:
            
            # slightly awkward sequence to only open finished files... (but believe me - it works!)
            
            fns = io.expand_files(args.filename)
            # print(fns)
            fns = [fn for fn in fns if fn not in seen_raw_files]
            # validation...
            try:
                fns = io.expand_files(fns, validate=not args.no_validate)
            except (OSError, IOError, RuntimeError) as err:
                print(f'Could not open file(s) {" ".join(fns)} because of', err)
                print('Possibly, it is still being written to. Waiting a bit...')
                sleep(5)
                continue
                
            if not fns:
                print('No new files, waiting a bit...')
                sleep(5)
                continue
            else:
                print(f'Found new files(s):\n', '\n'.join(fns))
                try:
                    ds_raw = Dataset.from_files(fns, chunking=args.chunksize)
                except Exception as err:
                    print(f'Could not open file(s) {" ".join(fns)} because of', err)
                    print('Possibly, it is still being written to. Waiting a bit...')
                    sleep(5)
                    continue
        
        else:
            fns = io.expand_files(args.filename, validate=not args.no_validate)
            if fns:
                ds_raw = Dataset.from_files(fns, chunking=args.chunksize)
            else:
                print(f'\n---\n\nFile(s) {args.filename} not found or (all of them) invalid.')
                return
            
        seen_raw_files.extend(ds_raw.files)
        
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
        os.makedirs(opts.proc_dir, exist_ok=True)
        ds_compute.init_files(overwrite=True)

        print('Storing meta tables...')
        ds_compute.store_tables(shots=True, features=True)

        print(f'Processing diffraction data... monitor progress at {client.dashboard_link} (or forward port if remote)')
        chunk_info = quick_proc(ds_compute, opts, label_raw, label, client)
        
        # make sure that the calculation went consistent with the data set
        for (sh, sh_grp), (ch, ch_grp) in zip(ds_compute.shots.groupby(['file', 'subset']), chunk_info.groupby(['file', 'subset'])):
            if any(sh_grp.shot_in_subset.values != np.sort(np.concatenate(ch_grp.shot_in_subset.values))):
                raise ValueError(f'Incosistency between calculated data and shot list in {sh[0]}: {sh[1]} found. Please investigate.')
    
        ds_compute.write_list(args.list_file, append = args.append)
        
        print(f'Computation done. Processed files are in {args.list_file}')
               
        if not args.wait_for_files:
            break

if __name__ == '__main__':
    main()