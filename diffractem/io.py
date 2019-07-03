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


def expand_files(file_list: Union[str, list], scan_shots=False):
    if isinstance(file_list, list) or isinstance(file_list, tuple):
        fl = file_list
        if scan_shots:
            fl = pd.DataFrame(fl, columns=['file'])

    elif isinstance(file_list, str) and file_list.endswith('.lst'):
        if scan_shots:
            fl = pd.read_csv(file_list, sep=' ', header=None, engine='python',
                             names=['file', 'Event'])
            if fl.Event.isna().all():
                fl.drop('Event', axis=1, inplace=True)
        else:
            fl = []
            for s in open(file_list, 'r').readlines():
                if '//' in s:
                    raise RuntimeError('Shot identifier found in list file. You may want to set scan_shots=True')
                fl.append(s.split(' ', 1)[0].strip())

    elif isinstance(file_list, str) and (file_list.endswith('.h5') or file_list.endswith('.nxs')):
        fl = [file_list, ]
        if scan_shots:
            fl = pd.DataFrame(fl, columns=['file'])

    else:
        raise TypeError('file_list must be a list file, single h5/nxs file, or a list of filenames')

    if (not scan_shots) and (not len(fl) == len(set(fl))):
        raise ValueError('File identifiers are not unique, most likely because the file names are not.')

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
            dict_to_h5(grp.require_group(nk), v)
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
            d[k] = h5_to_dict(v)
        elif isinstance(v, h5py.Dataset):
            if (len(v.shape) > 0) and (len(v) > max_len):
                continue
            d[k] = v.value
    return d


def apply_shot_selection(lists, stacks, min_chunk=None, reset_shot_index=True):
    """
    Applies the selection of shots as defined by the 'selected' column of the shot list, returning corresponding
    subsets of both lists (pandas DataFrames) and stacks (dask arrays).
    :param lists: flat dict of lists. Does not handle subsets, so use flat=True when reading from file
    :param stacks: dict of arrays. Again, not accounting for subsets. Use flat=True for reading
    :param min_chunk: minimum chunk size of the output arrays along the stacked dimension
    :param reset_index: if True, the returned shot list has its index reset, with correspondingly updated serial numbers in peak list. Recommended.
    :return new_lists, new_stacks: subselected lists and stacks
    """
    # n
    shots = lists['shots']  # just a shortcut
    new_lists = lists.copy()
    new_lists['shots'] = lists['shots'].query('selected').copy()
    print('Keeping {} shots out of {}'.format(len(new_lists['shots']), len(shots)))
    if 'peaks' in lists.keys():
        # remove rejected shots from the peak list
        # TODO: why is this not simply done with a right merge?
        peaksel = lists['peaks'].merge(shots[['selected']], left_on='serial', right_index=True)['selected']
        new_lists['peaks'] = lists['peaks'].loc[peaksel, :]

        if reset_shot_index:
            new_lists['shots']['newEv'] = range(len(new_lists['shots']))
            new_lists['peaks'] = new_lists['peaks'].merge(new_lists['shots'].loc[:, ['newEv', ]],
                                                          left_on='serial', right_index=True)
            new_lists['peaks']['serial'] = new_lists['peaks']['newEv']
            new_lists['peaks'].drop('newEv', axis=1, inplace=True)
            new_lists['shots'].drop('newEv', axis=1, inplace=True)

    if reset_shot_index:
        new_lists['shots'].reset_index(drop=True, inplace=True)

    new_stacks = {}
    for k, stk in stacks.items():

        # select the proper images from the stack
        stack = stk[shots['selected'].values, ...]

        # if desired, re-chunk such that chunks don't become too small
        if min_chunk is not None:
            nchk = 0
            fchks = []
            for ii, chk in enumerate(stack.chunks[0]):
                nchk += chk
                if nchk >= min_chunk:
                    fchks.append(nchk)
                    nchk = 0
                elif ii == len(stack.chunks[0]) - 1:
                    fchks[-1] += nchk

            stack = stack.rechunk({0: tuple(fchks)})

        new_stacks.update({k: stack})

    return new_lists, new_stacks


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


def get_meta_lists(filename, base_path='/%/data', labels=None):
    warnings.warn('Please use get_meta_list instead if you know what you\'re looking for. It is WAY faster.',
        DeprecationWarning)
    fns = expand_files(filename)
    identifiers = base_path.rsplit('%', 1)
    lists = defaultdict(list)
    # print(fns)

    for fn in fns:
        # print(fn)
        with h5py.File(fn) as fh:

            if len(identifiers) == 1:
                base_grp = {'': fh[identifiers[0]]}
            else:
                base_grp = fh[identifiers[0]]
            # print(base_grp)
            for subset, ssgrp in base_grp.items():
                # print(list(ssgrp.keys()))
                if (len(identifiers) > 1) and identifiers[1]:
                    if identifiers[1].strip('/') in ssgrp.keys():
                        grp = ssgrp[identifiers[1].strip('/')]
                    else:
                        continue
                else:
                    grp = ssgrp  # subset identifier is on last level

                if isinstance(grp, h5py.Group):
                    # print(grp)
                    for tname, tgrp in grp.items():
                        # print(tname, tgrp)
                        if tgrp is None:
                            # can happen for dangling soft links
                            continue
                        if ((labels is None) or (tname in labels)) and ('table_type' in tgrp.attrs):
                            newlist = pd.read_hdf(fn, tgrp.name)
                            newlist['subset'] = subset
                            newlist['file'] = fn
                            # newlist['shot_in_subset'] = range(newlist.shape[0])

                            lists[tname].append(newlist)
                            # print(f'Appended {len(newlist)} items from {fn}: {subset} -> list {tname}')

    lists = {tn: pd.concat(t, axis=0, ignore_index=True) for tn, t in lists.items()}
    return lists


def store_data_stacks(filename, stacks, shots=None, base_path='/%/data', store_shots=True, **kwargs):
    if shots is None:
        print('No shot list provided; getting shots from data file(s).')
        shots = get_meta_lists(filename, base_path, 'shots')['shots']

    if filename is not None:
        fns = expand_files(filename)
    else:
        fns = shots['file'].unique()

    if 'serial' in shots.columns:
        serial = shots['serial'].values
    else:
        serial = shots.index.values

    if (np.diff(serial) != 1).any():
        warnings.warn('Serial numbers are not equally incrementing by one. Just saying.', RuntimeWarning)

    stacks.update({'serial': da.from_array(serial, chunks=(1,))})
    counters = {ln: 0 for ln in stacks.keys()}

    datasets = []
    arrays = []
    files = []

    try:
        for fn in fns:

            f = h5py.File(fn)
            files.append(f)
            fshots = shots.loc[shots['file'] == fn, :]

            for sn, stack in stacks.items():
                for subset, ssshots in fshots.groupby('subset'):
                    arr = stack[ssshots.index.values, ...]
                    path = base_path.replace('%', subset) + '/' + sn
                    ds = f.require_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                           chunks=tuple([c[0] for c in arr.chunks]), **kwargs)

                    arrays.append(arr)
                    datasets.append(ds)
                    counters[sn] += arr.shape[0]
                    print(f'{sn}: {subset}//{ssshots.index.values[0]}...{ssshots.index.values[-1]} -> {fn}:{path}')

        for k, v in stacks.items():
            if v.shape[0] != counters[k]:
                print(f'Warning: {counters[k]} out of {v.shape[0]} entries in stack {k} will be stored.')

        with warnings.catch_warnings():
            with dask.diagnostics.ProgressBar():
                da.store(arrays, datasets)

    except Exception as err:
        [f.close() for f in files]
        raise err

    if store_shots and shots is not None:
        store_meta_lists(filename, {'shots': shots}, base_path=base_path)
        print('Stored shot list.')


def get_data_stacks(filename, base_path='/%/data', labels=None):
    # Internally, this function is structured 99% as get_meta_lists, just operating on dask
    # arrays, not pandas frames
    fns = expand_files(filename)
    identifiers = base_path.rsplit('%', 1)
    stacks = defaultdict(list)

    for fn in fns:
        fh = h5py.File(fn)

        try:
            if len(identifiers) == 1:
                base_grp = {'': fh[identifiers[0]]}
            else:
                base_grp = fh[identifiers[0]]
            for subset, ssgrp in base_grp.items():

                if (len(identifiers) > 1) and identifiers[1]:
                    if identifiers[1].strip('/') in ssgrp.keys():
                        grp = ssgrp[identifiers[1].strip('/')]
                else:
                    grp = ssgrp  # subset identifier is on last level

                if isinstance(grp, h5py.Group):
                    for dsname, ds in grp.items():
                        if ds is None:
                            # can happen for dangling soft links
                            continue
                        if ((labels is None) or (dsname in labels)) \
                                and isinstance(ds, h5py.Dataset) \
                                and ('pandas_type' not in ds.attrs):
                            stacks[dsname].append(da.from_array(ds, chunks=ds.chunks))

        except Exception as err:
            fh.close()
            raise err

    stacks = {sn: da.concatenate(s, axis=0) for sn, s in stacks.items()}
    return stacks

def filter_shots(filename_in, filename_out, query, min_chunk=None, shots=None, list_args=None, stack_args=None):
    """
    Macro function to apply filtering operation to an entire HDF file containing metadata lists and data stacks.
    Shots are kept if the "selected" column in the shot list is True, and the query string (see below) is fulfilled.
    :param filename_in: input HDF file
    :param filename_out: output HDF file
    :param query: criterion for inclusion of shots in output file, written as a string which can contain columns of the
    shot list, and evaluates to a boolean. Shots evaluated as true are kept.
    Example: 'peak_count >= 50 and region == 13'
    :param min_chunk: minimum chunk size along stacked direction for the output stacks
    :param shots: optional. shot list that overwrites the one from the input file. Can be useful to skip an intermediate
    step when performing more complex subselections.
    :return: none
    """
    lists = get_meta_lists(filename_in, flat=False)
    if shots is not None:
        lists['shots'] = shots
    stacks = get_data_stacks(filename_in, flat=False)
    new_lists = {}
    new_stacks = {}

    for ssn, ssl in lists.items():

        if query is not None:

            try:
                ssl['shots'].loc[ssl['shots'].eval('not ({})'.format(query)), 'selected'] = False
            except Exception as err:
                print('Possibly you have used a column not present in the shot index in the query expression.')
                print('The columns are: {}'.format(ssl['shots'].columns.values))
                raise err

        sss = stacks[ssn]
        new_ssl, new_sss = apply_shot_selection(ssl, sss)
        new_lists.update({ssn: new_ssl})
        new_stacks.update({ssn: new_sss})

    if list_args is None:
        list_args = {}
    store_meta_lists(filename_out, new_lists, flat=False, **list_args)

    if stack_args is None:
        stack_args = {}
    store_data_stacks(filename_out, new_stacks, flat=False, **stack_args)
