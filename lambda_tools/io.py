import re
from glob import glob
import h5py
import numpy as np
import pandas as pd
from dask import array as da
import json
from collections import defaultdict
import dask.diagnostics
from io import StringIO
import os.path
from lambda_tools import normalize_names
from tifffile import imread


def get_files(file_list):

    if isinstance(file_list, list) or isinstance(file_list, tuple):
        fl = file_list

    elif isinstance(file_list, str) and file_list.endswith('.lst'):
        fl = [s.strip() for s in open(file_list, 'r').readlines()]

    elif isinstance(file_list, str) and (file_list.endswith('.h5') or file_list.endswith('.nxs')):
        fl = [file_list, ]

    else:
        raise TypeError('file_list must be a list file, single h5/nxs file, or a list of filenames')

    id = [fn.rsplit('.', 1)[0].rsplit('/', 1)[-1] for fn in fl]   # this defines the identifiers in the meta data.

    if not len(id) == len(set(id)):
        raise ValueError('File identifiers are not unique, most likely because the file names are not.')

    return fl, id

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


def meta_to_nxs(nxs_file, meta=None, exclude=('Detector',), meta_grp='/entry/instrument',
                data_grp='/entry/data', data_field='data', data_location='/entry/instrument/detector/data'):
    """
    Merges a dict containing metadata information for a serial data acquisition into an existing detector nxs file.
    Additionally, it adds a soft link to the actual data for easier retrieval later (typically into /entry/data)
    :param nxs_file:
    :param meta:
    :param exclude:
    :param meta_grp:
    :param data_grp:
    :param data_field:
    :param data_location:
    :return:
    """

    f = h5py.File(nxs_file, 'r+')

    if meta is None:
        meta = nxs_file.rsplit('.', 1)[0] + '.json'

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
        dgrp.attrs['NX_class'] = np.array('NXdata')
        dgrp.attrs['signal'] = np.array(data_field)

        if data_field in dgrp.keys():
            del dgrp[data_field]
        dgrp[data_field] = h5py.SoftLink(data_location)

    f.close()


def copy_h5(fn_from, fn_to, exclude=('%/detector/data', '/%/data/%'), mode='w-',
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

    if fn_from.rsplit('.', 1)[-1] == 'lst':
        old_files = [f.strip() for f in open(fn_from).readlines()]
        new_files = []

        for ofn in old_files:
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
                #print(f'Copying table {key}')
                to.copy(ds, key)

            elif isinstance(ds, h5py.Group):
                #print(f'Creating group {key}')
                new_grp = to.require_group(key)
                for k, v in ds.attrs.items():
                    #if
                    try:
                        new_grp.attrs.create(k, v)
                    except TypeError as err:
                        new_grp.attrs.create(k, np.string_(v))

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


def read_crystfel_stream(filename, serial_offset=-1):
    """
    Reads a crystfel stream file and returns the contained peaks and predictions
    :param filename: obvious, right?
    :param serial_offset: offset to be applied to the event serial numbers in the crystfel file. By default -1,
        as the crystfel serials are 1-based by default.
    :return: peak list, prediction list
    """
    import pandas as pd
    from io import StringIO
    with open(filename, 'r') as fh:
        fstr = StringIO(fh.read())
    event = -1
    shotnr = -1
    subset = ''
    serial = -1
    init_peak = False
    init_index = False
    linedat_peak = []
    linedat_index = []
    for ln, l in enumerate(fstr):

        # Event id and indexing scheme
        if 'Event:' in l:
            event = l.split(': ')[-1].strip()
            shotnr = int(event.split('//')[-1])
            subset = event.split('//')[0].strip()
            if not subset:
                subset = '(none)'
            #print(event, ':', shotnr, ':', subset)
            continue
        if 'Image filename:' in l:
            filename = l.split(':')[-1].strip()
        if 'Image serial number:' in l:
            serial = int(l.split(': ')[1]) + serial_offset
            continue
        if 'indexed_by' in l:
            indexer = l.split(' ')[2].replace("\n", "")
            continue

        # Information from the peak search
        if 'End of peak list' in l:
            init_peak = False
            continue

        if init_peak:
            #print(filename, event, serial, subset, shotnr)
            linedat_peak.append('{} {} {} {} {} {}'.format(l.strip(), filename, event, serial, subset, shotnr))

        if 'fs/px   ss/px (1/d)/nm^-1   Intensity  Panel' in l:  # 'Peaks from peak search' in l: #Placed after the writing, so that line are written from the next
            init_peak = True
            continue

        # Information from the indexing
        if 'Cell parameters' in l:
            a, b, c, dummy, al, be, ga = l.split(' ')[2:9]
            continue
        if 'astar' in l:
            astar_x, astar_y, astar_z = l.split(' ')[2:5]
            continue
        if 'bstar' in l:
            bstar_x, bstar_y, bstar_z = l.split(' ')[2:5]
            continue
        if 'cstar' in l:
            cstar_x, cstar_y, cstar_z = l.split(' ')[2:5]
            continue
        if 'predict_refine/det_shift' in l:
            xshift = l.split(' ')[3]
            yshift = l.split(' ')[6]
            continue

        if 'End of reflections' in l:
            init_index = False
            continue

        if init_index:
            recurrent_info = [filename, event, serial, subset, shotnr, indexer, a, b, c, al, be, ga, astar_x, astar_y, astar_z, bstar_x, bstar_y,
                              bstar_z, cstar_x, cstar_y, cstar_z]  # List with recurrent info
            recurrent_info = ' '.join(
                map(str, recurrent_info))  # Transform list of several element in a list of a single string
            linedat_index.append('{} {}'.format(l.strip(), recurrent_info))

        if 'h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel' in l:  # Placed after the writing, so that line are written from the next
            init_index = True
            continue

    df_peak = pd.read_csv(StringIO('\n'.join(linedat_peak)), delim_whitespace=True, header=None,
                          names=['fs/px', 'ss/px', '(1/d)/nm^-1', 'Intensity', 'Panel', 'File', 'Event', 'serial', 'subset', 'shot_in_subset']
                          ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(drop=True).drop('index',axis=1)
    df_index = pd.read_csv(StringIO('\n'.join(linedat_index)), delim_whitespace=True, header=None,
                           names=['h', 'k', 'l', 'I', 'Sigma(I)', 'Peak', 'Background', 'fs/px', 'ss/px', 'Panel', 'File',
                                  'Event', 'serial', 'subset', 'shot_in_subset', 'Indexer', 'a', 'b', 'c', 'Alpha', 'Beta', 'Gamma',
                                  'astar_x', 'astar_y', 'astar_z', 'bstar_x', 'bstar_y', 'bstar_z', 'cstar_x',
                                  'cstar_y', 'cstar_z']
                           ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(drop=True).drop('index',axis=1)

    return df_peak, df_index


def write_nxds_spots(peaks, filename='SPOT.nXDS', prefix='diffdat_?????', threshold=0,
                     pixels=958496, min_pixels=3):

    fstr = StringIO()
    fstr.write(prefix + '.h5\n')

    peaks['nXDS_panel'] = 1
    jj = 0
    for ii, grp in peaks.groupby('Event'):
        if len(grp) < threshold:
            continue
        fstr.write('{}\n'.format(ii + 1)) # nXDS is one-based!
        fstr.write('{} {} {}\n'.format(pixels - min_pixels * len(grp), min_pixels * len(grp), len(grp)))
        grp.loc[:, ['nXDS_panel', 'fs/px', 'ss/px', 'Intensity']].to_csv(fstr, header=False, sep=' ', index=False)
        jj += 1
    with open(filename, 'w') as fh:
        fh.write(fstr.getvalue())

    peaks.drop('nXDS_panel', axis=1, inplace=True)


def get_raw_stack(shots_or_files, sort_by=('subset', 'region', 'run', 'crystal_id', 'frame'),
                   drop_invalid=True, aggregate=None, agg_by=('subset', 'region', 'run', 'crystal_id'),
                  max_chunk=16384, min_chunk=None,
                   data_path='/entry/instrument/detector/data',**kwargs):

    # TODO: documentation
    # TODO: check nxs file for dropped frames and other anomalies

    sort_by = list(sort_by)
    agg_by = list(agg_by)

    if isinstance(shots_or_files, str) \
            or isinstance(shots_or_files, list) \
            or isinstance(shots_or_files, tuple):
        shot_list = build_shot_list(shots_or_files, **kwargs)
    else:
        assert isinstance(shots_or_files, pd.DataFrame)
        shot_list = shots_or_files

    if drop_invalid:
        valid = (shot_list[['region', 'crystal_id', 'run', 'frame', 'shot']] >= 0).all(axis=1)
        shot_list_final = shot_list.loc[valid, :].copy()
    else:
        shot_list_final = shot_list.copy()

    if 'selected' in shot_list_final.columns:
        shot_list_final = shot_list_final.loc[shot_list_final['selected'],:]

    if sort_by is not None:
        shot_list_final.sort_values(sort_by, inplace=True)

    # In the following, some magic is done to get the optimal chunking for all following operations.
    # file_block in each of the dataframes is an ID of continuous blocks that will remain together
    # when applying the sorting and selection of images to the data stack. file_block is montonically
    # increasing, and does not necessarily identify the same block in shot_list and shot_list_final
    block_change = (shot_list_final['file'] != shot_list_final['file'].shift(1)) | \
        (shot_list_final['shot'].diff() != 1)
    shot_list_final['file_block'] = block_change.astype(int).cumsum() - 1

    shot_list = shot_list.merge(shot_list_final[['file_block', ]], left_index=True, right_index=True, how='outer').fillna(-1)
    block_change2 = shot_list['file_block'] != shot_list['file_block'].shift(1)
    shot_list['file_block'] = block_change2.astype(int).cumsum() - 1

    shot_list_final.reset_index(drop=True, inplace=True)

    stacks = {}

    # read the image stacks from all nxs files as dask arrays, chunking them according to file_block
    for _, fn in shot_list_final['file'].drop_duplicates().items():
        print(f'Reading raw file {fn}')
        fh = h5py.File(fn + '.nxs')
        ds = fh[data_path]
        cs0 = shot_list.groupby(['file','file_block']).size().loc[fn].values
        for ii in np.where(cs0 > max_chunk)[0][::-1]:
            # limit the chunk size
            dm = divmod(cs0[ii], max_chunk)
            cs0 = np.insert(np.delete(cs0, ii), ii, np.append(np.repeat(max_chunk, dm[0]), dm[1]))
        stacks[fn] = da.from_array(ds, chunks=(tuple(cs0), ds.chunks[1], ds.chunks[2]))

    imgs = []

    # now go file_block-wise through the final shot list and append the corresponding ranges of images.
    # as we have done the chunking according to the same file_block structure, this operation does not
    # change any chunks (which would be expensive)
    for _, shdat in shot_list_final.groupby('file_block'):
        img_block = stacks[shdat['file'].iloc[0]][shdat['shot'].values,...]
        imgs.append(img_block)
    stack = da.concatenate(imgs)

    # now aggregate the shots if desired by summing or averaging
    if aggregate is not None:
        agg_stack = []
        agg_shots = []
        for nm, grp in shot_list_final.groupby(agg_by):
            if aggregate in 'sum':
                agg_stack.append(stack[grp.index.values, ...].sum(axis=0, keepdims=True))
            if aggregate in ['mean', 'average', 'avg']:
                agg_stack.append(stack[grp.index.values, ...].mean(axis=0, keepdims=True))
            agg_shots.append(grp.iloc[0, :])  # at this point, they all should be the same

        shot_list_final = pd.DataFrame(agg_shots).reset_index(drop=True)
        stack = da.concatenate(agg_stack)

    # Finally, optionally re-chunk the stack such that all chunks are at least min_chunk large,
    # respecting the chunk boundaries as they are
    if min_chunk is not None:
        nchk = 0
        fchks = []
        for ii, chk in enumerate(stack.chunks[0]):
            nchk += chk
            if nchk >= min_chunk:
                fchks.append(nchk)
                nchk = 0
            elif ii == len(stack.chunks[0])-1:
                fchks[-1] += nchk

        stack = stack.rechunk({0: tuple(fchks)})

    return stack, shot_list_final


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
#n
    shots = lists['shots']  # just a shortcut
    new_lists = lists.copy()
    new_lists['shots'] = lists['shots'].query('selected').copy()
    print('Keeping {} shots out of {}'.format(len(new_lists['shots']),len(shots)))
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

    fns, ids = get_files(file_list)

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

    fns, ids = get_files(filename)

    for fn, id in zip(fns, ids):
        fh = h5py.File(fn)  # otherwise file-open-trouble between pandas and h5py.
        fh.close()

        with pd.HDFStore(fn) as store:
            for ln, l in lists.items():
                for ssn, ssl in l.loc[l['file'] == id, :].groupby('subset'):
                    path = base_path.replace('%', ssn) + '/' + ln
                    try:
                        # subset and file are auto-generated when reloading
                        store.put(path, ssl.drop(['subset', 'file'], axis=1), format='table', data_columns=True, **kwargs)
                    except ValueError as err:
                        # most likely, the column titles contain something not compatible with h5 data columns
                        store.put(path, ssl.drop(['subset', 'file'], axis=1), format='table', **kwargs)


def get_meta_lists(filename, base_path='/%/data', labels=None):

    fns, ids = get_files(filename)
    identifiers = base_path.rsplit('%', 1)
    lists = defaultdict([])

    for fn, id in zip(fns, ids):

        fh = h5py.File(fn)

        try:
            base_grp = fh[identifiers[0]]

            for subset, ssgrp in base_grp.items():

                if identifiers[1]:
                    grp = ssgrp[identifiers[1].strip('/')]
                else:
                    grp = ssgrp     # subset identifier is on last level

                if isinstance(grp, h5py.Group):
                    for tname, tgrp in grp.items():
                        if ((labels is None) or (tname in labels)) and ('table_type' in tgrp.attrs):
                            newlist = pd.read_hdf(fn, tgrp.name)
                            newlist['subset'] = subset
                            newlist['file'] = id
                            lists[tname].append(newlist)

            lists = {tn: pd.concat(t, axis=0, ignore_index=True) for tn, t in lists.items()}

        finally:
            fh.close()

    return lists



def store_data_stacks(filename, stacks, flat=True, shots=None, base_path='/entry/data/%', **kwargs):
    """
    Compute and store dask arrays into HDF file, using the standard structure
    :param filename: Name of HDF file to store arrays into
    :param stacks: dict of dask arrays. Either nested dict with top-level keys corresponding to subsets and list names
    in sub-level keys, or flat dict with subset names separately supplied through shots list
    :param flat: if True, flat dict structure (see above) is assumed. shots must be set then.
    :param shots: shot list corresponding to the stacks, containing "subset" column
    :param kwargs: forwarded to dask.array.to_hdf5
    :return: nothing
    """

    if filename.rsplit('.', 1)[-1] == 'lst':
        fn2 = filename.rsplit('.', 1)[0] + '_temp.h5'
        # unfortunately, the usual master-h5 trick does not work because dask's to_hdf5 does not handle
        # external links properly. still using a similar mechanism, to enforce consistency.
        try:
            with h5py.File(make_master_h5(filename, fn2)) as f:
                subsets = list(f.keys())
                links = [f.get(k, getlink=True) for k in subsets]
                ssadd = {s: l for s, l in zip(subsets, links)}
        finally:
            os.remove(fn2)

        datasets = []
        arrays = []
        files =[]
        for subset, idcs in shots.groupby('subset').indices.items():
            lnk = ssadd[subset]
            f = h5py.File(lnk.filename)
            files.append(f)

            for sn, s in stacks.items():
                arr = s[idcs,...]
                path = base_path.replace('/%', lnk.path) + '/' + sn
                ds = f.require_dataset(path, shape=arr.shape, dtype=arr.dtype, **kwargs)
                arrays.append(arr)
                datasets.append(ds)

                print(f'Storing stack {sn} for subset {subset} into {lnk.filename} -> {path}')

        with dask.diagnostics.ProgressBar():
            da.store(arrays, datasets)

        [f.close() for f in files]

        return

    allstacks = {}

    if flat:
        for sn, s in stacks.items():
            if shots is None:
                raise ValueError('When using a flat dict, you have to supply a shot list with subset column')
            for subset, idcs in shots.groupby('subset').indices.items():
                allstacks.update({
                    base_path.replace('%', '{}').format(subset) + '/' + sn: s[idcs, ...]})

    else:
        for subset, ssss in stacks.items():
            for sn, sss in ssss.items():
                allstacks.update({
                    base_path.replace('%', '{}').format(subset) + '/' + sn: sss})

    print('Computing and storing the following stacks: ')
    for k, v in allstacks.items():
        print(k, '\t', v)

    with dask.diagnostics.ProgressBar():
        #print(kwargs)
        da.to_hdf5(filename, allstacks, **kwargs)


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
        #print('Writing array to: ' + meta_path)
        da.to_hdf5(filename, {meta_path: darr}, **kwargs)
        if 'subset' in shots.columns:
            store_meta_lists(filename, {listname: shots}, flat=True)
        else:
            store_meta_lists(filename, {subset_label: {listname: shots}}, flat=False)

    return label_string, meta_path



def get_nxs_list(filename, what='shots'):
    if what == 'shots':
        return get_meta_lists(filename, True, '/%/data', ['shots'])['shots']
    if what in ['crystals', 'features']:
        return get_meta_lists(filename, True, '/%/map', [what])[what]
    if what in ['peaks', 'predict']:
        return get_meta_lists(filename, True, '/%/results', ['peaks'])['predict']
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
        store_meta_lists(filename, {'shots': list}, True, '/%/data')
    elif what in ['crystals', 'features']:
        store_meta_lists(filename, {'features': list}, True, '/%/map')
    elif what in ['peaks', 'predict']:
        store_meta_lists(filename, {'features': list}, True, '/%/results')
    else:
        raise ValueError('Unknown list type')


def get_data_stacks(filename, flat=True, base_path='/entry/data/%', labels=None):
    """
    Reads all data arrays from a (processed) HDF file as dask dataframes and returns them as a nested dictionary.
    :param filename: Name of the HDF data file from serial acquisition.
    :param flat: if True, all subset data is concatenated for each stack. Order is alphabetical according to the
    subset names.
    :return: dict of the structure {'subset1_name': {'raw': raw_data, 'bgcorr': bg_corrected, ...},
    'subset2_name": .....} etc. if flat=False.
    """

    if filename.rsplit('.', 1)[-1] == 'lst':
        # this only works with new-style nxs files...
        try:
            fn2 = filename.rsplit('.', 1)[0] + '_temp.h5'
            fn2 = make_master_h5(filename, fn2)
            stacks = get_data_stacks(fn2, flat, base_path, labels)
        finally:
            os.remove(fn2)
        return stacks

    identifiers = base_path.rsplit('%', 1)
    fh = h5py.File(filename)
    base_grp = fh[identifiers[0]]

    stacks = {}

    for subset, ssgrp in base_grp.items():

        if identifiers[1]:
            grp = ssgrp[identifiers[1].strip('/')]
        else:
            # subset identifier is on last level
            grp = ssgrp

        if isinstance(grp, h5py.Group):
            stacks.update({subset: {}})
            for dslabel, ds in grp.items():
                if ((labels is None) or (dslabel in labels)) \
                        and isinstance(ds, h5py.Dataset) and ('pandas_type' not in ds.attrs):
                    stacks[subset].update({dslabel: da.from_array(ds, ds.chunks)})

    fh.close()

    if flat:
        stks = defaultdict(list)
        for sn, sd in sorted(stacks.items()):
            for tn, td in sd.items():
                stks[tn].append(td)
        stacks = {k: da.concatenate(v) for k, v in stks.items()}

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
            fullpath = '{}/{}/{}/{}'.format(prefix,path['subset'],array_name,path[array_name])
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

    if not (isinstance(filenames, list) or isinstance (filenames, tuple)):
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
        #px = meta['Scanning']['Pixel size']['x']
        #py = meta['Scanning']['Pixel size']['y']

        if nshots != meta['Scanning']['Total Pts']:
            raise ValueError('Image stack size does not match scan.')

        npos = int(nshots/frames)

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
            crystal_id = np.repeat(-1,npos)

        if use_coords:
            coords = np.loadtxt(basename + coord_postfix)
            coords = np.append(coords, [[np.nan, np.nan]], axis=0) # required to allow -1 coordinate
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
    #return pd.concat(shots).reset_index(drop=True)