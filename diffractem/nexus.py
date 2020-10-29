import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_EXCEPTION
from itertools import repeat
from typing import Union, List, Tuple, Optional
import os
import h5py
import numpy as np
import pandas as pd
from warnings import warn
from .io import expand_files, dict_to_h5


def _get_table_from_single_file(fn: str, path: str) -> pd.DataFrame:
    identifiers = path.rsplit('%', 1)
    lists = []
    with h5py.File(fn, 'r') as fh:

        try:
            if len(identifiers) == 1:
                subsets = ['']
            else:
                subsets = fh[identifiers[0]].keys()

            for subset in subsets:
                tbl_path = path.replace('%', subset)
                if tbl_path not in fh:
                    # warn(f'Group {tbl_path} not found in {fn}.')
                    raise KeyError(f'Group {tbl_path} not found in {fn}.')
                    # newlist = None
                    
                if 'pandas_type' in fh[tbl_path].attrs:
                    # print(f'Found list {tbl_path} in Pandas/PyTables format')
                    newlist = pd.read_hdf(fn, tbl_path)
                else:
                    dt = {}
                    for key, val in fh[tbl_path].items():
                        if val.ndim != 1:
                            warn('Data fields in list group must be 1-D, {} is {}-D. Skipping.'.format(key, val.ndim))
                            continue
                        dt_field = val.dtype
                        if 'label' in val.attrs:
                            k = val.attrs['label']
                        else:
                            k = key
                        if dt_field.type == np.string_:
                            try:
                                dt[k] = val[:].astype(np.str)
                            except UnicodeDecodeError as err:
                                print(f'Field {key} of type {dt_field} gave decoding trouble:')
                                raise err
                        else:
                            dt[k] = val[:]              
                    newlist = pd.DataFrame().from_dict(dt)

                newlist['subset'] = subset
                newlist['file'] = fn
                lists.append(newlist)
                
        except KeyError as kerr:
            raise KeyError(f'{path} not found in {fn}.')

        return pd.concat(lists, axis=0, ignore_index=True)


def get_table(files: Union[list, str], path='/%/shots', parallel=True) -> pd.DataFrame:

    files = expand_files(files)

    if parallel:
        with ProcessPoolExecutor() as p:
            out = p.map(_get_table_from_single_file, files, repeat(path))
            # ftrs = []
            # for fn in files:
            #     ftrs.append(p.submit(_get_table_from_single_file, fn, path))
            # TODO make this more robust against errors by changing to submit instead of map and handling single-file errors
                
    else:
        out = map(_get_table_from_single_file, files, repeat(path))
        
    out = pd.concat(out, ignore_index=True, sort=False)

    return out


def _store_table_to_single_subset(tbl: pd.DataFrame, fn: str, path: str, subset: str, format: str = 'nexus'):
    """
    Helper function. Internal use only.
    """

    tbl_path = path.replace('%', subset)
    if format == 'table':
        try:
            tbl.to_hdf(fn, tbl_path, format='table', data_columns=True)
        except ValueError:
            tbl.to_hdf(fn, tbl_path, format='table')

    elif format == 'nexus':
        with h5py.File(fn, 'a') as fh:
            for key, val in tbl.iteritems():
                #print(f'Storing {key} ({val.shape}, {val.dtype}) to {fn}: {path}')
                grp = fh.require_group(tbl_path)
                grp.attrs['NX_class'] = 'NXcollection'
                k = key.replace('/', '_').replace('.', ' ')
                try:
                    if k not in grp:
                        ds = grp.require_dataset(k, shape=val.shape, dtype=val.dtype, maxshape=(None,))
                    else:
                        ds = grp[k]
                        if ds.shape[0] != val.shape[0]:
                            ds.resize(val.shape[0], axis=0)
                            #print('resizing', k)
                    ds[:] = val
                except (TypeError, OSError) as err:
                    if val.dtype == 'O':                        
                        val2 = val.astype('S')
                        if k in grp:
                            del grp[k]
                        ds = grp.require_dataset(k, shape=val.shape, dtype=val2.dtype, maxshape=(None,))
                        ds[:] = val2
                    else:
                        raise err

                ds.attrs['label'] = key
    else:
        raise ValueError('Storage format must be "table" or "nexus".')


def store_table(table: pd.DataFrame, path: str, 
                parallel: bool = True, format: str = 'nexus',
                file: Optional[str] = None, subset: Optional[str] = None):
    """
    Stores a pandas DataFrame containing 'file' and 'subset' columns to multiple HDF5 files. Essentially a
    multi-file, multi-processed wrapper to pd.to_hdf
    :param table: DataFrame to be stored
    :param path: path in HDF5 files. % will be substituted by the respective subset name
    :param parallel: if True (default), writes files in parallel
    :param format: can be 'nexus' to write columns of table in separate arrays, or 'tables' to use PyTables to write
            a HDF5 table object.
    :return: list of futures (see documentation of concurrent.futures). [None] if parallel=False
    """

    # TODO: could be that parallel execution with multiple subsets/table/types will not work

    if (file is None) and parallel:

        with ProcessPoolExecutor() as exec:
            futures = []
            try:
                for (fn, ssn), ssdat in table.groupby(['file', 'subset']):
                    futures.append(exec.submit(_store_table_to_single_subset, ssdat, fn, path, ssn, format))
            except Exception as err:
                print('Error during storing table in', path)
                print('Table columns are:', ', '.join(table.columns))
                # print(table)
                raise err

            wait(futures, return_when=FIRST_EXCEPTION)

            for f in futures:
                if f.exception():
                    raise f.exception()

            return futures

    else:
        #print(path)
        #print(table.columns)

        if file is not None:
            _store_table_to_single_subset(table, file, path, subset, format)

        else:
            for (fn, ssn), ssdat in table.groupby(['file', 'subset']):
                _store_table_to_single_subset(ssdat, fn, path, ssn, format)

        return [None]

def _save_single_chunk(dat: np.ndarray, file: str, subset: str, label: str, 
                       idcs: Union[list, np.ndarray], data_pattern: str, lock):   
    lock.acquire()
    with h5py.File(file, 'a') as fh:
        path = f'{data_pattern}/{label}'.replace('%', subset)
        fh[path][idcs,:,:] = dat
    lock.release()
    return file, subset, path, idcs

def _save_single_chunk_multi(chks: dict, file: str, subset: str, 
                       idcs: Union[list, np.ndarray], lock):   
    lock.acquire()
    with h5py.File(file, 'a') as fh:
        for p, d in chks.items():
            fh[p.replace('%', subset)][idcs,...] = d
    lock.release()
    return file, subset, list(chks.keys()), idcs

def meta_to_nxs(filename, meta=None, exclude=('Detector',), meta_grp='/entry/instrument',
                data_grp='/entry/data', data_field='raw_counts', data_location='/entry/instrument/detector/data'):
    """
    Merges a dict containing metadata information for a serial data acquisition into an existing detector nxs file.
    Additionally, it adds a soft link to the actual data for easier retrieval later (typically into /entry/data)
    :param filename: NeXus file or lists
    :param meta: can be set to {} -> no meta action performed. Or a JSON file name. If None, a JSON file name will be
        derived from nxs_file by replacing .nxs by .json (useful in loops)
    :param exclude: names of meta groups or fields to exclude
    :param meta_grp: location in the NeXus, where the metadata should go to
    :param data_grp: location of softlink to the data stack. No softlink action if None.
    :param data_field: name of the softlink to the data stack
    :param data_location: location of the data stack
    :return:
    """

    # TODO: add functions to include flat field and pixel mask

    if (not isinstance(filename, str)) or filename.endswith('.lst'):
        fns = expand_files(filename)
        for fn in fns:
            meta_to_nxs(fn, meta=meta, exclude=exclude, meta_grp=meta_grp,
                        data_grp=data_grp, data_field=data_field, data_location=data_location)
        return

    with h5py.File(filename, 'r+') as f:

        if meta is None:
            meta = filename.rsplit('.', 1)[0] + '.json'

        if isinstance(meta, str):
            try:
                meta = json.load(open(meta))
            except FileNotFoundError:
                print('No metafile found.')
                meta = {}

        elif isinstance(meta, dict):
            pass

        elif isinstance(meta, pd.DataFrame):
            meta = next(iter(meta.to_dict('index').values()))

        dict_to_h5(f.require_group(meta_grp), meta, exclude=exclude)

        if data_grp is not None:
            dgrp = f.require_group(data_grp)
            dgrp.attrs['NX_class'] = np.string_('NXdata')
            dgrp.attrs['signal'] = np.string_(data_field)

            if data_field in dgrp.keys():
                del dgrp[data_field]
            dgrp[data_field] = h5py.SoftLink(data_location)


def get_meta_fields(files: Union[str, list], dataset_paths: Union[list, str, tuple, dict], shorten_labels=True):
    """
    Get arbitrary meta data from files.
    :param files:
    :param dataset_paths: list of dataset paths, or dict of structure {dataset: default value}
    :param shorten_labels: only use final section of labels for columns of returned DataFrame
    :return: pandas DataFrame of metadata
    """

    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    if isinstance(dataset_paths, list) or isinstance(dataset_paths, tuple):
        dataset_paths = {f: None for f in dataset_paths}

    values = defaultdict(dict)
    dtypes = {}
    fns = expand_files(files)

    for fn in fns:
        with h5py.File(fn, mode='r') as fh:
            for field, default in dataset_paths.items():

                identifiers = field.rsplit('%', 1)

                if len(identifiers) == 1:
                    subsets = ['']
                else:
                    subsets = fh[identifiers[0]].keys()

                for subset in subsets:
                    try:
                        # print(f[field])
                        values[field][(fn, subset)] = fh[field.replace('%', subset)][...]
                        dtypes[field] = fh[field.replace('%', subset)].dtype
                        if dtypes[field] == 'O':
                            dtypes[field] = str
                        # print(field, fh[field.replace('%', subset)].dtype)
                    except KeyError:
                        values[field][(fn, subset)] = default

    newcols = {'level_0': 'file', 'level_1': 'subset'}
    if shorten_labels:
        newcols.update({k: k.rsplit('/', 1)[-1] for k in dataset_paths})
    return pd.DataFrame(values).astype(dtypes).reset_index().rename(columns=newcols)


def copy_h5(fn_from, fn_to, exclude=('%/detector/data', '/%/data/%', '/%/results/%'), mode='w-',
            print_skipped=False, h5_folder=None, h5_suffix='.h5'):
    """
    Copies datasets h5/nxs files or lists of them to new ones, with exclusion of datasets.
    :param fn_from: single h5/nxs file or list file
    :param fn_to: new file name, or new list file. If the latter, specify with h5_folder and h5_suffix how the new names
        are supposed to be constructed
    :param exclude: patterns for data sets to be excluded. All regular expressions are allowed, % is mapped to .*
        (i.e., any string of any length), for compatibility with CrystFEL
    :param mode: mode in which new files are opened. By default w-, i.e., files are created, but never overwritten
    :param print_skipped: print the skipped data sets, for debugging
    :param h5_folder: if operating on a list: folder where new h5 files should go
    :param h5_suffix: if operating on a list: suffix appended to old files (after stripping their extension)
    :return:
    """

    # multi-file copy, using recursive call.
    if (isinstance(fn_from, str) and fn_from.endswith('.lst')) or isinstance(fn_from, list):
        warn('Calling copy_h5 on a file list is not recommended anymore', DeprecationWarning)
        old_files = expand_files(fn_from)
        new_files = []

        for ofn in old_files:
            # print(ofn)
            # this loop could beautifully be parallelized. For later...
            if h5_folder is None:
                h5_folder = ofn.rsplit('/', 1)[0]
            if h5_suffix is None:
                h5_suffix = ofn.rsplit('.', 1)[-1]
            nfn = h5_folder + '/' + ofn.rsplit('.', 1)[0].rsplit('/', 1)[-1] + h5_suffix
            new_files.append(nfn)
            # exclude detector data and shot list
            copy_h5(ofn, nfn, exclude, mode, print_skipped)

        with open(fn_to, 'w') as f:
            f.write('\n'.join(new_files))

        return

    # single-file copy
    try:

        # no exclusion... simply copy file
        if len(exclude) == 0:
            from shutil import copyfile
            copyfile(fn_from, fn_to)
            return

        exclude_regex = [re.compile(ex.replace('%', '.*')) for ex in exclude]

        def copy_exclude(key, ds, to):
            # function to copy a single entry within a HDF hierarchy, and do recursive calls
            # if required. If it finds its key in the exclusion patterns, just skips that entry.

            for ek in exclude_regex:
                if ek.fullmatch(ds.name) is not None:
                    if print_skipped:
                        print(f'Skipping key {key} due to {ek}')
                    return

            if isinstance(ds, h5py.Dataset):
                to.copy(ds, key)

            elif isinstance(ds, h5py.Group) and 'table_type' in ds.attrs.keys():
                # pandas table is a group. Do NOT traverse into it (or experience infinite pain)
                # print(f'Copying table {key}')
                to.copy(ds, key)

            elif isinstance(ds, h5py.Group):
                # print(f'Creating group {key}')
                new_grp = to.require_group(key)

                # attribute copying. Lots of error catching required.
                try:
                    for k, v in ds.attrs.items():
                        try:
                            new_grp.attrs.create(k, v)
                        except TypeError as err:
                            new_grp.attrs.create(k, np.string_(v))
                except OSError:
                    # some newer HDF5 attribute types (used by pytables) will crash h5py even just listing them
                    # print(f'Could not copy attributes of group {ds.name}')
                    pass

                for k, v in ds.items():
                    lnk = ds.get(k, getlink=True)
                    if isinstance(lnk, h5py.SoftLink):
                        for ek in exclude_regex:
                            if ek.fullmatch(lnk.path) is not None:
                                if print_skipped:
                                    print(f'Skipping soft link to {ek}')
                                break
                        else:
                            new_grp[k] = h5py.SoftLink(lnk.path)
                        continue

                    copy_exclude(k, v, new_grp)

                # for k, v in ds.items():
                #     lnk = ds.get(k, getlink=True)
                #     if isinstance(lnk, h5py.SoftLink):
                #         new_grp[k] = h5py.SoftLink(lnk.path)
                #         continue
                #     copy_exclude(k, v, new_grp)

        with h5py.File(fn_from, mode='r') as f, h5py.File(fn_to, mode=mode) as f2:
            copy_exclude('/', f, f2)

    except Exception as err:
        if os.path.exists(fn_to):
            os.remove(fn_to)
        print(f'Error occurred while attempting to copy data from {fn_from} to {fn_to}.')
        raise err
