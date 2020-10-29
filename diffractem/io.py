import h5py
import numpy as np
import pandas as pd
from dask import array as da
from collections import defaultdict
import dask.diagnostics
import os.path
from diffractem import normalize_names
import warnings
from typing import Union
from glob import glob


def expand_files(file_list: Union[str, list], scan_shots=False, validate=False):

    def remove_bs(fns):
        return [fn.replace('\\', '/') for fn in fns]
    
    if isinstance(file_list, list) or isinstance(file_list, tuple):
        fl = remove_bs(file_list)
        if scan_shots:
            fl = pd.DataFrame(fl, columns=['file'])

    elif isinstance(file_list, str) and file_list.endswith('.lst'):
        if scan_shots:
            fl = pd.read_csv(file_list, sep=' ', header=None, engine='python',
                             names=['file', 'Event'])
            fl['file'] = remove_bs(fl['file'])
            if fl.Event.isna().all():
                fl.drop('Event', axis=1, inplace=True)
        else:
            fl = []
            for s in open(file_list, 'r').readlines():
                if '//' in s:
                    raise RuntimeError('Shot identifier found in list file. You may want to set scan_shots=True')
                fl.append(s.split(' ', 1)[0].strip())
            fl = remove_bs(fl)

    elif isinstance(file_list, str) and (file_list.endswith('.h5') or file_list.endswith('.nxs')):
        fl = remove_bs(sorted(glob(file_list)))
        if scan_shots:
            fl = pd.DataFrame(fl, columns=['file'])

    else:
        raise TypeError('file_list must be a list file, single or glob pattern of h5/nxs files, or a list of filenames')

    if (not scan_shots) and (not len(fl) == len(set(fl))):
        raise ValueError('File identifiers are not unique, most likely because the file names are not.')
        
    if validate:
        if scan_shots:
            raise ValueError('Validation is only allowed if scan_shot=False.')
        valid_files = []
        for r in fl:
            try:
                with h5py.File(r, 'r') as fh:
                    
                    for k in fh.keys():
                    
                        if (f'/{k}/shots' in fh) and (f'/{k}/map/features' in fh) and (f'/{k}/data' in fh):
                            # print(r,': file validated!')
                            valid_files.append(r)
                        else:
                            print(r, k, ': invalid file/subset!')       
            except (OSError, IOError) as err:
                print('Could not open file', r, 'for validation because:')
                print(err)
                    
        return valid_files

    else:
        return fl


def dict_to_h5(grp, data, exclude=()):
    """
    Write dictionary into HDF group (or file) object
    :param grp: HDF group or file object
    :param data: dictionary to be written into HDF5
    :param exclude: dataset or group names to be excluded
    :return:
    """
    for k, v in data.items():
        nk = normalize_names(k)
        if k in exclude:
            continue
        elif isinstance(v, dict):
            dict_to_h5(grp.require_group(nk), v, exclude=exclude)
        else:
            if nk in grp.keys():
                grp[nk][...] = v
            else:
                grp.create_dataset(nk, data=v)


def h5_to_dict(grp, exclude=('data', 'image'), max_len=100):
    """
    Get dictionary from HDF group (or file) object
    :param grp: HDF group or file
    :param exclude: (sub-)group or dataset names to be excluded; by default 'data' and 'image
    :param max_len: maximum length of data field to be included (along first direction)
    :return: dictionary corresponding to HDF group
    """
    d = {}
    for k, v in grp.items():
        if k in exclude:
            continue
        if isinstance(v, h5py.Group):
            d[k] = h5_to_dict(v, exclude=exclude, max_len=max_len)
        elif isinstance(v, h5py.Dataset):
            if (len(v.shape) > 0) and (len(v) > max_len):
                print('Skipping', v.shape, len(v), max_len, v)
                continue
            d[k] = v.value
    return d

def make_master_h5(file_list, file_name=None, abs_path=False, local_group='/',
                   remote_group='/entry', verbose=False):
    fns, ids = expand_files(file_list, True)

    if isinstance(file_list, str) and file_list.endswith('.lst'):
        if file_name is None:
            file_name = file_list.rsplit('.', 1)[0] + '.h5'
    else:
        if file_name is None:
            raise ValueError('Please provide output file name explicitly, if input is not a file list.')

    f = h5py.File(file_name, 'w')

    try:

        subsets = []

        for fn, id in zip(fns, ids):

            subset = id

            if subset in subsets:
                raise KeyError('File names are not unique!')
            else:
                subsets.append(subset)

            if abs_path:
                fn2 = os.getcwd() + '/' + fn
            else:
                fn2 = fn

            if not os.path.isfile(fn2):
                raise FileNotFoundError(f'File {fn2} present in {file_list} not found!')

            if verbose:
                print(f'Referencing file {fn2} as {subset}')
            if local_group != '/':
                f.require_group(local_group)

            f[local_group + '/' + subset] = h5py.ExternalLink(fn2, remote_group)

    except Exception as err:
        f.close()
        os.remove(file_name)
        raise err

    f.close()

    return file_name

