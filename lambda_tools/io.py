import re
from glob import glob
import h5py
import numpy as np
import pandas as pd
from dask import array as da
import json
from collections import defaultdict
import dask.diagnostics
import os.path
from lambda_tools import normalize_names, stream_parser
from tifffile import imread
import warnings


def read_crystfel_stream(filename, serial_start=-1):
    # emulates old behavior
    sfp = stream_parser.StreamParser(filename, serial_start)
    pk = sfp._peaks.merge(sfp._shots[['file', 'Event', 'subset', 'shot_in_subset']], on=['file', 'Event'], how='left')
    pred = sfp._indexed.merge(sfp._shots, on=['file', 'Event', 'serial'], how='left')
    return pk, pred


def get_files(file_list, scan_shots=False):
    if isinstance(file_list, list) or isinstance(file_list, tuple):
        fl = file_list

    elif isinstance(file_list, str) and file_list.endswith('.lst'):
        if scan_shots:
            fl = pd.read_csv(file_list, sep=' |//', header=None, engine='python',
                             names=['file', 'subset', 'shot_in_subset'])
            if fl.subset.isna().all():
                fl.drop('subset', axis=1, inplace=True)
            if fl.shot_in_subset.isna().all():
                fl.drop('shot_in_subset', axis=1, inplace=True)
        else:
            fl = []
            for s in open(file_list, 'r').readlines():
                if '//' in s:
                    raise RuntimeError('Shot identifier found in list file. You may want to set scan_shots=True')
                fl.append(s.split(' ', 1)[0].strip())

    elif isinstance(file_list, str) and (file_list.endswith('.h5') or file_list.endswith('.nxs')):
        fl = [file_list, ]

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


def meta_from_nxs(filename, datasets):
    """
    Get arbitrary meta data from files. So far does _not_ handle % placeholder for subsets because I'm lazy.
    :param filename:
    :param datasets: list of dataset paths, or dict of structure {dataset: default value}
    :return:
    """

    if isinstance(datasets, str):
        datasets = [datasets]

    if isinstance(datasets, list) or isinstance(datasets, tuple):
        datasets = {f: None for f in datasets}

    datasets = {k.replace('%', 'entry'): v for k, v in datasets.items()}

    values = defaultdict(dict)
    fns = get_files(filename)
    for fn in fns:
        with h5py.File(fn) as f:
            for field, default in datasets.items():
                try:
                    # print(f[field])
                    values[field][fn] = f[field][...]
                except KeyError:
                    values[field][fn] = default

    return dict(values)


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
        fns = get_files(filename)
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

    if (isinstance(fn_from, str) and fn_from.endswith('.lst')) or isinstance(fn_from, list):
        old_files = get_files(fn_from)
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

    try:

        if len(exclude) == 0:
            from shutil import copyfile
            copyfile(fn_from, fn_to)

        f = h5py.File(fn_from)
        f2 = h5py.File(fn_to, mode)

        exclude_regex = [re.compile(ex.replace('%', '.*')) for ex in exclude]

        def copy_exclude(key, ds, to):

            for ek in exclude_regex:
                if ek.fullmatch(ds.name) is not None:
                    if print_skipped:
                        print(f'Skipping key {key} due to {ek}')
                    return

            if isinstance(ds, h5py.Dataset):
                to.copy(ds, key)

            elif isinstance(ds, h5py.Group) and 'table_type' in ds.attrs.keys():
                # pandas table is a group. Do NOT traverse into it or pain
                # print(f'Copying table {key}')
                to.copy(ds, key)

            elif isinstance(ds, h5py.Group):
                # print(f'Creating group {key}')
                new_grp = to.require_group(key)
                # attribute copying is nasty. Lots of error catching required.
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
                        new_grp[k] = h5py.SoftLink(lnk.path)
                        continue
                    copy_exclude(k, v, new_grp)

        copy_exclude('/', f, f2)

    except Exception as err:
        f.close()
        f2.close()
        os.remove(fn_to)
        raise err

    f.close()
    f2.close()


def modify_stack(filename, shot_list=None, base_path='/%/data', labels='raw_counts', sort_by=None,
                 drop_invalid=True, aggregate=None, agg_by=('file', 'subset', 'region', 'run', 'crystal_id'),
                 newchunk=None):
    # TODO: documentation
    # TODO: check nxs file for dropped frames and other anomalies

    if shot_list is None:
        shot_list = get_meta_lists(filename, base_path, 'shots')['shots']

    if drop_invalid:
        valid = (shot_list[['region', 'crystal_id', 'run', 'frame', 'shot']] >= 0).all(axis=1)
        shot_list_final = shot_list.loc[valid, :].copy()
    else:
        shot_list_final = shot_list.copy()

    if 'selected' in shot_list_final.columns:
        shot_list_final = shot_list_final.loc[shot_list_final['selected'], :]

    if sort_by is not None:
        shot_list_final.sort_values(list(sort_by), inplace=True)

    print(f'Initial: {shot_list.shape[0]} shots, Selection: {shot_list_final.shape[0]} shots')

    stacks = get_data_stacks(filename, base_path, labels)
    print(f'Loaded raw stacks: \n{stacks}.')

    # now aggregate the shots if desired, by summing or averaging
    if aggregate is not None:
        agg_stacks = {k: [] for k in stacks.keys()}
        agg_shots = []
        for _, grp in shot_list_final.groupby(list(agg_by)):
            for sn, s in stacks.items():
                if aggregate in 'sum':
                    newstack = s[grp.index.values, ...].sum(axis=0, keepdims=True)
                elif aggregate in ['mean', 'average', 'avg']:
                    newstack = s[grp.index.values, ...].mean(axis=0, keepdims=True)
                else:
                    raise ValueError('aggregate must be sum or mean')
                if newchunk is not None:
                    # TODO this does not seem to do anything? Why?!
                    newstack = da.rechunk(newstack, (newchunk, -1, -1))

                agg_stacks[sn].append(newstack)
                agg_shots.append(grp.iloc[[0], :])  # at this point, they all should be the same

        shot_list_final = pd.concat(agg_shots, ignore_index=True)
        print(f'Aggregated: {shot_list_final.shape[0]} shots')

        stacks_final = {sn: da.concatenate(s) for sn, s in agg_stacks.items()}

    else:
        stacks_final = {sn: s[shot_list_final.index.values, ...] for sn, s in stacks.items()}
        shot_list_final.reset_index(drop=True, inplace=True)

    shot_list_final['shot_in_subset'] = shot_list_final.groupby(['file', 'subset']).cumcount()

    return stacks_final, shot_list_final


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
    fns, ids = get_files(file_list, True)

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


def store_meta_lists(filename, lists, base_path='/%/data', **kwargs):
    if filename is not None:
        fns = get_files(filename)
    else:
        fns = pd.concat([l['file'] for l in lists.values()], ignore_index=True).unique()

    # print(fns)
    counters = {ln: 0 for ln in lists.keys()}

    for fn in fns:
        fh = h5py.File(fn)  # otherwise file-open-trouble between pandas and h5py.
        fh.close()

        with pd.HDFStore(fn) as store:
            for ln, l in lists.items():
                for ssn, ssl in l.loc[l['file'] == fn, :].groupby('subset'):
                    path = base_path.replace('%', ssn) + '/' + ln
                    try:
                        # subset and file are auto-generated when reloading
                        store.put(path, ssl.drop(['subset', 'file'], axis=1).reset_index(drop=True),
                                  format='table', data_columns=True, **kwargs)
                    except ValueError as err:
                        # most likely, the column titles contain something not compatible with h5 data columns
                        store.put(path, ssl.drop(['subset', 'file'], axis=1).reset_index(drop=True),
                                  format='table', **kwargs)
                    counters[ln] += ssl.shape[0]

    for k, v in lists.items():
        if v.shape[0] != counters[k]:
            print(f'Warning: {counters[k]} out of {v.shape[0]} entries in list {k} were stored.')


def get_meta_lists(filename, base_path='/%/data', labels=None):
    fns = get_files(filename)
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


def store_meta_array(filename, array_label, identifier, array, shots=None, listname=None,
                     subset_label=None, base_path='entry/meta', chunks=None,
                     simulate=False, **kwargs):
    if listname is None:
        listname = 'shots'

    if shots is None:
        shots = get_meta_lists(filename, flat=True)[listname]

    if not subset_label:
        if 'subset' in shots.columns:
            labels = shots['subset'].drop_duplicates()
            if len(labels) > 1:
                raise ValueError('If shot/crystal list has more than one subset, a ' +
                                 'subset label must be given to associate the meta array with.')
            subset_label = labels.iloc[0]

    label_string = ''
    query_string = ''

    for k, v in identifier.items():
        label_string = label_string + '{}_{}_'.format(k, v)
        query_string = query_string + '{} == {} and '.format(k, v)

    label_string = label_string[:-1]

    if 'subset' in shots.columns:
        shots.loc[shots.eval('{}subset == \'{}\''.format(query_string, subset_label)), array_label] = label_string
    else:
        shots.loc[shots.eval('{}True'.format(query_string)), array_label] = label_string

    meta_path = base_path + '/' + subset_label + '/' + array_label + '/' + label_string

    if simulate:
        print(meta_path)
        return label_string, meta_path

    else:
        if chunks is None:
            chunks = array.shape
        darr = da.from_array(array, chunks)
        # print('Writing array to: ' + meta_path)
        da.to_hdf5(filename, {meta_path: darr}, **kwargs)
        if 'subset' in shots.columns:
            store_meta_lists(filename, {listname: shots}, flat=True)
        else:
            store_meta_lists(filename, {subset_label: {listname: shots}}, flat=False)

    return label_string, meta_path


def get_nxs_list(filename, what='shots'):
    if what == 'shots':
        shots = get_meta_lists(filename, '/%/data', ['shots'])['shots']
        if 'shot_in_subset' not in shots:
            shots['shot_in_subset'] = shots.groupby(['file', 'subset']).cumcount()
        return shots
    if what in ['crystals', 'features']:
        return get_meta_lists(filename, '/%/map', [what])[what]
    if what in ['peaks', 'predict']:
        return get_meta_lists(filename, '/%/results', [what])[what]
    # for later: if none of the usual ones, just crawl the file for something matching
    return None


def store_nxs_list(filename, list, what='shots'):
    """
    Store a pandas frame into a nxs-compliant file following the SerialED convention, which is practically defined
    in this function...
    :param filename:
    :param list: pandas data frame
    :param what: possible values at the moment: shots, crystals, features, peaks, predict
    :return:
    """
    if what == 'shots':
        store_meta_lists(filename, {'shots': list}, '/%/data')
    elif what in ['crystals', 'features']:
        store_meta_lists(filename, {'features': list}, '/%/map')
    elif what in ['peaks', 'predict']:
        store_meta_lists(filename, {what: list}, '/%/results')
    else:
        raise ValueError('Unknown list type')


def store_data_stacks(filename, stacks, shots=None, base_path='/%/data', store_shots=True, **kwargs):
    if shots is None:
        print('No shot list provided; getting shots from data file(s).')
        shots = get_meta_lists(filename, base_path, 'shots')['shots']

    if filename is not None:
        fns = get_files(filename)
    else:
        fns = shots['file'].unique()

    if 'serial' in shots.columns:
        serial = shots['serial'].values
    else:
        serial = shots.index.values

    if (np.diff(serial) != 1).any():
        raise RuntimeWarning('Serial numbers are not equally incrementing by one! Danger!')

    stacks.update({'serial': da.from_array(serial.reshape(-1, 1), chunks=(1, -1))})
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
    fns = get_files(filename)
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


def get_meta_array(filename, array_label, shot, subset=None, base_path='/entry/meta'):
    """

    :param filename:
    :param array_label:
    :param shot:
    :return:
    """

    if isinstance(shot, pd.Series):
        pass
    elif isinstance(shot, pd.DataFrame):
        print('DataFrame (list) has been passed as shot, only the first selected line will be used to fetch array!')
        shot = shot.loc[shot['selected'].values, :].iloc[0, :]
    else:
        raise TypeError('Shot argument must be pandas Series (or DataFrame, but only first entry will be used!')

    if 'subset' in shot.keys():
        subset = shot['subset']

    pathname = base_path + '/' + subset + '/' + array_label + '/' + shot[array_label]
    print('Fetching array in: ' + pathname)

    with h5py.File(filename) as fh:
        array = fh[pathname][:]

    return array


def copy_meta_array(fn_from, fn_to, shots, array_name='stem', prefix='/entry/meta'):
    with h5py.File(fn_from) as fh1, h5py.File(fn_to) as fh2:
        for _, path in shots[['subset', array_name]].drop_duplicates().iterrows():
            fullpath = '{}/{}/{}/{}'.format(prefix, path['subset'], array_name, path[array_name])
            print('Copying ' + fullpath)
            fh2[fullpath] = fh1[fullpath][:]


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


def build_shot_list(filenames, use_region_id=True, use_mask=True, use_coords=True, subset_label=None,
                    mask_postfix='_coll_mask.tif', coord_postfix='_foc_coord.txt',
                    region_id_pos=-2):
    """
    Builds a list of shots contained in the given filenames, collecting information from associated mask files and
    coordinate lists. Returned lists contain information like crystal ID, coordinates, region etc. in the exact
    same order as contained in the files given in the input. This should be the first step in all analysis.
    :param filenames:
    :param use_region_id:
    :param use_mask:
    :param use_coords:
    :param subset_label:
    :param mask_postfix:
    :param coord_postfix:
    :param region_id_pos:
    :return:
    """

    if not (isinstance(filenames, list) or isinstance(filenames, tuple)):
        filenames = glob(filenames)
        filenames = filenames.sort()

    shots = []

    for fidx, fn in enumerate(filenames):
        basename = fn.rsplit('.', 1)[0]
        run = int(re.findall('\d+', fn)[-1])

        meta = json.load(open(basename + '.json'))
        nshots = meta['Detector']['FrameNumbers']
        frames = meta['Scanning']['Parameters']['Smp/Pos']
        nx = meta['Scanning']['Parameters']['Line (X)']['ROI len']
        ny = meta['Scanning']['Parameters']['Frame (Y)']['ROI len']
        sx = meta['Scanning']['Parameters']['Line (X)']['ROI st']
        sy = meta['Scanning']['Parameters']['Frame (Y)']['ROI st']
        # px = meta['Scanning']['Pixel size']['x']
        # py = meta['Scanning']['Pixel size']['y']

        if nshots != meta['Scanning']['Total Pts']:
            raise ValueError('Image stack size does not match scan.')

        npos = int(nshots / frames)

        if use_region_id:
            region = int(re.findall('\d+', fn)[region_id_pos])
        else:
            region = fidx

        if use_mask:
            mask = imread(basename + mask_postfix).astype(np.int64)
            crystal_id = mask[mask != 0].ravel(order='F')
            if len(crystal_id) != npos:
                raise ValueError('Marked pixel number in mask does not match image stack.')
        else:
            crystal_id = np.repeat(-1, npos)

        if use_coords:
            coords = np.loadtxt(basename + coord_postfix)
            coords = np.append(coords, [[np.nan, np.nan]], axis=0)  # required to allow -1 coordinate
            crystal_x = coords[crystal_id, 1]
            crystal_y = coords[crystal_id, 0]
        else:
            crystal_x = np.ones(len(crystal_id)) * np.nan
            crystal_y = np.ones(len(crystal_id)) * np.nan

        xrng = sx + np.arange(nx)
        yrng = sy + np.arange(ny)
        X, Y = np.meshgrid(xrng, yrng)

        if use_mask:
            pos_x = X[mask != 0]
            pos_y = Y[mask != 0]
        else:
            pos_x = X.ravel(order='F')
            pos_y = Y.ravel(order='F')

        alldat = {'region': np.repeat(region, len(crystal_id)),
                  'run': np.repeat(run, len(crystal_id)),
                  'file': np.repeat(basename, len(crystal_id)),
                  'crystal_id': crystal_id,
                  'crystal_x': crystal_x,
                  'crystal_y': crystal_y,
                  'pos_x': pos_x,
                  'pos_y': pos_y}

        alldat = {k: np.repeat(v, frames, 0) for k, v in alldat.items()}
        alldat['shot'] = np.arange(nshots)
        alldat['frame'] = np.tile(np.arange(frames), npos)
        alldat['selected'] = True
        shots.append(pd.DataFrame(alldat))

    shots = pd.concat(shots).reset_index(drop=True)

    if subset_label is not None:
        shots['subset'] = subset_label

    return shots
    # return pd.concat(shots).reset_index(drop=True)
