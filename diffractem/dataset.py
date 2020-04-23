# dedicated to Thea and The Penguin

import pandas as pd
import numpy as np
import dask.array.gufunc
from dask import array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from . import io, nexus
from .stream_parser import StreamParser
from .map_image import MapImage
import h5py
from typing import Union, Dict, Optional, List, Tuple, Callable
import copy
from collections import defaultdict
from warnings import warn, catch_warnings, simplefilter
from tables import NaturalNameWarning
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
from contextlib import contextmanager
import os
from numba import jit

# top-level helper functions for chunking operations
# ...to be refactored into tools or compute later...
def _check_commensurate(init: Union[list, tuple, np.ndarray], final: Union[list, tuple, np.ndarray], 
                        equal_size: bool = False):
    '''check if blocks with sizes in init are commensurate with (i.e. have boundaries aligned with)
    blocks in final, and (optionally) if final blocks in final are equally-sized within each block in initial.
    Useful to check if a dask rechunk operation will act across boundaries of existing chunks,
    which is often something you'll want to try to avoid (and might be a sign that something is going wrong).
    Blocks in final must hence be smaller than those in init, i.e. len(final) >= len(init), 
    and of course: sum(final) == sum(init).
    Returns whether the blocks are commensurate, and (if so), the number of  
    final blocks in each of the initial block.'''
    #TODO consider using numba jit

    final_inv = list(final)[::-1] # invert for faster popping
    init = list(init)
    if sum(init) != sum(final):
        raise ValueError('Sum of init and final must be identical.')
    blocksize = []
    if equal_size:
        for s0 in init:
            # iterate over initial blocks
            n_final_in_initial = s0 // final_inv[-1]
            for _ in range(n_final_in_initial):
                # iterate over final blocks within initial
                if (s0 % final_inv.pop()) != 0:
                    return False, None
            blocksize.append(n_final_in_initial)
    else:
        for s0 in init:
            # iterate over initial blocks
            # n_rem = copy.copy(s0)
            n_rem = s0
            b_num = 0
            while n_rem != 0:
                n_rem -= final_inv.pop()
                b_num += 1
                if n_rem < 0:
                    # incommensurate block found!
                    return False, None
            blocksize.append(b_num)
    assert len(final_inv) == 0
    return True, blocksize

def _agg_groups(stack: np.ndarray, labels: Union[np.ndarray, list, tuple], agg_function: callable, *args, **kwargs):
    '''Apply aggregating function to a numpy stack group-by-group, with groups defined by unique labels,
    and return the concatenated results; i.e., the length of the result along the aggregation
    axis equals the number of unique labels.
    '''

    res_list = []
    labels = labels.squeeze()
    for lbl in np.unique(labels):
        res_list.append(agg_function(stack[labels == lbl,...], *args, **kwargs))
    return np.concatenate(res_list)

def _map_sub_blocks(stack: da.Array, labels: Union[np.ndarray, list, tuple], func: callable, aggregating: bool = True,
                     *args, **kwargs):
    '''Wrapper for da.map_blocks, which instead of applying the function chunk-by-chunk can apply it also to sub-groups
    within each chunk, as identified by unique labels (e.g. integers). Useful if you want to use large chunks to have fast computation, but
    want to apply the function to smaller blocks. Obvious example: you want to sum frames from a diffraction
    movie, but have many diffraction movies stored in each single chunk, as otherwise the chunk number would be too large.
    The input stack must be chunked along its 0th axis only, and len(labels) must equal the height of the stack. 
    If aggregating=True, func is assumed to reduce the sub-block height to 1 (like summing all stack frames), whereas
    aggregating=False assumes func to leave the sub-block sizes as is (e.g. for cumulative summing).'''

    chunked_labels = da.from_array(labels.reshape((-1,1,1)), chunks=(stack.chunks[0],-1,-1), name='sub_block_label')
    cc_out = _check_commensurate(stack.chunks[0], np.unique(labels, return_counts=True)[1], equal_size=False)
    if not cc_out[0]:
        raise ValueError('Mismatched chunk structure: mapping groups are not within single chunk each')
    if 'chunks' in kwargs:
        final_chunks = kwargs['chunks']
    else:
        final_chunks = (tuple(cc_out[1]), ) + stack.chunks[1:] if aggregating else stack.chunks
    return da.map_blocks(_agg_groups, stack, chunked_labels, 
        agg_function=func, chunks=final_chunks, *args, **kwargs)

class Dataset:

    def __init__(self):

        self._shots_changed = False
        self._peaks_changed = False
        self._predict_changed = False
        self._features_changed = False

        # HDF5 file addresses
        self.data_pattern = '/%/data'
        self.shots_pattern = '/%/shots'
        self._fallback_shots_pattern = '/%/data/shots'
        self.result_pattern = '/%/results'
        self.map_pattern = '/%/map'
        self.instrument_pattern = '/%/instrument'
        self.parallel_io = True

        # internal stuff
        self._file_handles = {}
        self._stacks = {}
        self._shot_id_cols = ['file', 'Event']
        self._feature_id_cols = ['crystal_id', 'region', 'sample']
        self._stacks_open = False

        # tables
        self._shots = pd.DataFrame(columns=self._shot_id_cols + self._feature_id_cols + ['selected'])
        self._peaks = pd.DataFrame(columns=self._shot_id_cols)
        self._predict = pd.DataFrame(columns=self._shot_id_cols)
        self._features = pd.DataFrame(columns=self._feature_id_cols + ['file', 'subset'])

    def __str__(self):
        return (f'diffractem Dataset object spanning {len(self._shots.file.unique())} NeXus/HDF5 files\n-----\n'
                f'{self._shots.shape[0]} shots ({self._shots.selected.sum()} selected)\n'
                f'{self._peaks.shape[0]} peaks, {self._predict.shape[0]} predictions, '
                f'{self._features.shape[0]} features\n'
                f'{len(self._stacks)} data stacks: {", ".join(self._stacks.keys())}\n'
                f'Data stacks open: {self._stacks_open}\n'
                f'Data files open: {self._files_open}\n'
                f'Data files writable: {self._files_writable}')

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        # allows to access stacks with dot notation
        if attr == '_stacks':
            raise AttributeError()  # needed for copying the object to avoid infinite recursion
        if attr in self._stacks.keys():
            return self._stacks[attr]
        else:
            raise AttributeError(f'{attr} is neither a dataset attribute, nor a stack name.')

    @property
    def _files_open(self):
        return all([isinstance(f, h5py.File) for f in self.file_handles.values()])
    
    @property
    def _files_writable(self):
        return self._files_open and all([f.mode != 'r' for f in self.file_handles.values()])
    
    @property
    def _stack_in_memory(self):
        return {sn: len(stk.dask) == np.product(stk.numblocks) for sn, stk in self._stacks.items()}

    @property
    def file_handles(self):
        # handles to HDF5 files, if open. Otherwise None.
        return {fn: self._file_handles[fn] if fn in self._file_handles else None for fn in self.files}

    @property
    def stacks(self):
        if self._stacks_open:
            return self._stacks
        else:
            raise RuntimeError('Data stacks are not open!')

    @property
    def files(self):
        return list(self._shots['file'].unique())

    @property
    def shots(self):
        return self._shots

    @shots.setter
    def shots(self, value: pd.DataFrame):
        if (value.index != self._shots.index).any():
            raise ValueError('Shot index is different from existing one. Use modify_shots to change index.')
        if (value[self._shot_id_cols] != self._shots[self._shot_id_cols]).any().any():
            raise ValueError('Shot ID columns are different from existing ones. Use modify_shots to change index.')
        self._shots = value
        self._shots_changed = True

    @property
    def predict(self):
        warn('The prediction table functionality will likely be removed.', DeprecationWarning)
        return self._predict

    @property
    def features(self):
        return self._features

    @property
    def peaks(self):
        warn('The peak table functionality will likely be removed.', DeprecationWarning)
        return self._peaks
    
    @property
    def zchunks(self):
        # z chunks of dataset stacks
        allchk = [stk.chunks[0] for stk in self.stacks.values()]
        if allchk and all([chk == allchk[0] for chk in allchk]):
            return allchk[0]
        else:
            return None        

    @peaks.setter
    def peaks(self, value):
        warn('The peak table functionality will likely be removed.', DeprecationWarning)
        self._peaks = value
        self._peaks_changed = True

    @predict.setter
    def predict(self, value):
        warn('The prediction table functionality will likely be removed.', DeprecationWarning)
        self._predict = value
        self._predict_changed = True

    @features.setter
    def features(self, value):
        self._features = value
        self._features_changed = True

    @classmethod
    def from_files(cls, files: Union[list, str, tuple], init_stacks=True, load_tables=True, diff_stack_label='raw_counts', **kwargs):
        """
        Creates a data set from:
            * a .lst file name, which contains a simple list of H5 files (on separate lines). If the .lst file has CrystFEL-style
                event indicators in it, it will be loaded, and the events present in the list will be selected, the others not.
            * a glob pattern (like: 'data/*.h5')
            * a python iterable of files. 
            * a simple HDF5 file path
        :param files: see above
        :param init_stacks: initialize stacks, that is, briefly open the data stacks, check their lengths, and close
                them again. Does not hurt usually.
        :param load_tables: load the additional tables stored in the files (features, peaks, predictions)
        :param diff_stack_label: name of stack to be used for generating the shot table, if it's not stored in the files
        :param **kwargs: Dataset attributes to be set right away
        :return: dataset object
        """

        file_list = io.expand_files(files, scan_shots=True)
        # print(list(file_list))
        self = cls()

        for k, v in kwargs.items():
            self.__dict__[k] = v

        if len(file_list) == 1:
            print('Single-file dataset, disabling parallel I/O.')
            self.parallel_io = False

        self.load_tables(shots=True, files=list(file_list.file.unique()))
        if self.shots.shape[0] == 0:
            self.init_shot_table(file_list['file'], stack_label=diff_stack_label)

        # now set selected property...
        if 'Event' in file_list.columns:
            self._shots['selected'] = self._shots[self._shot_id_cols].isin(file_list[self._shot_id_cols]).all(axis=1)

        # and initialize stacks and tables
        if init_stacks:
            self.init_stacks()
        if load_tables:
            self.load_tables(features=True, peaks=True, predict=True)

        return self
    
    from_list = from_files # for compatibility

    def init_shot_table(self, files: list, stack_label: str = 'raw_counts'):
        identifiers = self.data_pattern.rsplit('%', 1)
        shots = []
        for fn in files:
            with h5py.File(fn, 'r') as fh:

                if len(identifiers) == 1:
                    subsets = ['']
                else:
                    subsets = fh[identifiers[0]].keys()

                file_shots = []
                for subset in subsets:
                    tbl_path = self.shots_pattern.replace('%', subset)
                    stk_path = (self.data_pattern + '/' + stack_label).replace('%', subset)
                    
                    stk_height = fh[stk_path].shape[0]
                    sss = pd.DataFrame(range(stk_height), columns=['shot_in_subset'])
                    sss['subset'] = subset
                    sss['file'] = fn
                    sss['Event'] = subset + '//' + sss['shot_in_subset'].astype(str)
                    sss['frame'] = 0
                    sss['selected'] = True
                    file_shots.append(sss)
                    
            shots.append(pd.concat(file_shots, axis=0).reset_index(drop=True))

        self._shots = pd.concat(shots, axis=0).reset_index(drop=True)
        print(f'Found {self.shots.shape[0]} shots, initialized shot table.')

    def load_tables(self, shots=False, features=False, peaks=False, predict=False, files=None):
        """
        Load pandas metadata tables from the HDF5 files. Set the argument for the table you want to load to True.
        :param shots: shot table
        :param features: feature table
        :param peaks: peak table
        :param predict: prediction table
        :param files: ...allows to supply a custom file list, instead of the stored one. Dangerous.
        :return:
        """

        if files is None:
            files = self.files

        if shots:
            if len(self.shots) > 0:
                warn('You are reloading the shot table. This can be dangerous. If you want to ensure a consistent'
                     ' data set, use the from_list class method instead, or start from an empty dataset.')
            try:
                try:
                    self._shots = nexus.get_table(files, self.shots_pattern,
                                                  parallel=self.parallel_io).reset_index(drop=True)
                except KeyError:
                    self._shots = nexus.get_table(files, self._fallback_shots_pattern,
                                                  parallel=self.parallel_io).reset_index(drop=True)

                self._shots_changed = False

                if 'shot_in_subset' not in self._shots.columns:
                    if 'shot' in self._shots.columns:
                        # seems to be a raw file from acquisition...
                        self._shots.rename(columns={'shot': 'shot_in_subset'}, inplace=True)
                    else:
                        self._shots['shot_in_subset'] = self._shots.groupby(['file', 'subset']).cumcount()

                if 'Event' not in self._shots.columns:
                    self._shots['Event'] = self._shots.subset + '//' + self._shots.shot_in_subset.astype(str)

                if 'selected' not in self._shots.columns:
                    self._shots['selected'] = True

                if 'stem' in self._shots.columns:
                    # deprecated....
                    self._shots.drop('stem', axis=1, inplace=True)

            except KeyError:
                print('No shots found at ' + self.shots_pattern)

        if features:
            try:
                self._features = nexus.get_table(files, self.map_pattern + '/features', parallel=self.parallel_io)
                self._features_changed = False

                if 'sample' not in self._features.columns:
                    sdat = nexus.get_meta_fields(list(self._features.file.unique()),
                                                 ['/%/sample/name', '/%/sample/region_id', '/%/sample/run_id']). \
                        rename(columns={'name': 'sample', 'region_id': 'region', 'run_id': 'run'})
                    self._features = self._features.merge(sdat, how='left', on=['file', 'subset'])

                self._features.drop_duplicates(self._feature_id_cols, inplace=True)

            except KeyError:
                print('No features found at ' + self.map_pattern + '/features')

        if peaks:
            try:
                self._peaks = nexus.get_table(files, self.result_pattern + '/peaks', parallel=self.parallel_io)
                self._peaks_changed = False
            except KeyError:
                pass
                #print('No peaks found at ' + self.result_pattern + '/peaks')

        if predict:
            try:
                self._predict = nexus.get_table(files, self.result_pattern + '/predict', parallel=self.parallel_io)
                self._predict_changed = False
            except KeyError:
                pass
                #print('No predictions found at ' + self.result_pattern + '/predict')

    def store_tables(self, shots: Union[None, bool] = None, features: Union[None, bool] = None,
                     peaks: Union[None, bool] = None, predict: Union[None, bool] = None, format: str = 'nexus'):
        """
        Stores the metadata tables (shots, features, peaks, predictions) into HDF5 files. For each of the tables,
        it can be automatically determined if they have changed and should be stored...
        HOWEVER, this only works if no inplace changes have been made. So don't rely on it too much.
        :param shots: True -> store shot list, False -> don't store shot list, None -> only store if changed
        :param features: similar
        :param peaks: similar
        :param predict: similar
        :param format: format to write metadata tables. 'nexus' (recommended) or 'tables' (old-style)
        :return:
        """
        fs = []

        if self._stacks_open and (format == 'tables'):
            warn('Data stacks are open, and will be transiently closed. You will need to re-create derived stacks.',
                 RuntimeWarning)
            stacks_were_open = True
            self.close_stacks()
        else:
            stacks_were_open = False

        simplefilter('ignore', NaturalNameWarning)
        if (shots is None and self._shots_changed) or shots:
            # sh = self.shots.drop(['Event', 'shot_in_subset'], axis=1)
            # sh['id'] = sh[['sample', 'region', 'run', 'crystal_id']].apply(lambda x: '//'.join(x.astype(str)), axis=1)
            fs.extend(nexus.store_table(self.shots, self.shots_pattern, parallel=self.parallel_io, format=format))
            self._shots_changed = False

        if (features is None and self._features_changed) or features:
            fs.extend(nexus.store_table(self.features, self.map_pattern + '/features', parallel=self.parallel_io,
                                        format=format))
            self._features_changed = False

        if (peaks is None and self._peaks_changed) or peaks:
            fs.extend(
                nexus.store_table(self.peaks, self.result_pattern + '/peaks', parallel=self.parallel_io, format=format))
            self._peaks_changed = False

        if (predict is None and self._predict_changed) or predict:
            fs.extend(nexus.store_table(self.predict, self.result_pattern + '/predict', parallel=self.parallel_io,
                                        format=format))
            self._predict_changed = False

        if stacks_were_open:
            self.open_stacks()

    def merge_stream(self, streamfile: Union[StreamParser, str]):
        """
        Loads a CrystFEL stream file, and merges its contents into the dataset object.
        :param streamfile: file name of streamfile, or StreamParser object
        :return:
        """

        # ...it would be way more elegant, to just associate a StreamParser object, and merge the list in
        # accessors. But the merges can become pretty slow for large files, so we do it only here.

        if isinstance(streamfile, str):
            stream = StreamParser(streamfile)
        else:
            stream = streamfile

        cols = list(self.shots.columns.difference(stream.shots.columns)) + self._shot_id_cols + ['subset',
                                                                                                 'shot_in_subset']
        self.shots = self.shots[cols].merge(stream.shots,
                                            on=self._shot_id_cols + ['subset', 'shot_in_subset'], how='left',
                                            validate='1:1')
        self.shots['selected'] = self.shots['serial'].notna()
        self.peaks = stream.peaks.merge(self.shots[self._shot_id_cols + ['subset', 'shot_in_subset']],
                                        on=self._shot_id_cols, how='inner')
        self.predict = stream.indexed.merge(self.shots[self._shot_id_cols + ['subset', 'shot_in_subset']],
                                            on=self._shot_id_cols, how='inner')

    def get_map(self, file, subset='entry') -> MapImage:
        # TODO: get a MapImage from stored data, with tables filled in from dataset
        raise NotImplementedError('does not work yet, sorry.')

    def _sel(self, obj: Union[None, pd.DataFrame, da.Array, np.array, h5py.Dataset, list, dict] = None):
        """
        General method to pick items that belong to selected shots from many kinds of data types.
        - For DataFrames, it matches the selected items by the datasets ID columns (usually 'file' and 'Event',
        or 'crystal_id' and 'region')
        - For anything slicable (dask or numpy array), it picks elements along the first array dimension,
        assuming that the stack is ordered the same way as the shot list.
        - Also accepts lists or dicts of all such objects and returns a corresponding list or dict.
        :param obj: DataFrame, numpy Array, dask Array, h5py Dataset, list, dict
        :return: same as obj
        """
        if obj is None:
            return self._shots.loc[self._shots.selected, :]
        elif isinstance(obj, pd.DataFrame) and all(c in obj.columns for c in self._shot_id_cols):
            return obj.merge(self._shots.loc[self._shots.selected, self._shot_id_cols],
                             on=self._shot_id_cols, how='inner', validate='m:1')
        elif isinstance(obj, pd.DataFrame) and all(c in obj.columns for c in self._feature_id_cols):
            return obj.merge(self._shots[self._feature_id_cols],
                             on=self._feature_id_cols, how='inner', validate='1:m')
        elif isinstance(obj, pd.DataFrame):
            raise ValueError(
                f'DataFrame must contain the columns {self._shot_id_cols} or {self._feature_id_cols}')
        elif isinstance(obj, list):
            return [self._sel(o) for o in obj]
        elif isinstance(obj, dict):
            return {k: self._sel(v) for k, v in obj.items()}
        else:
            return obj[self._shots.selected.values, ...]

    def select(self, query: str = 'True'):
        """
        Sets the 'selected' column of the shot list by a string query (eg. 'num_peaks > 30 and frame == 1').
        See pandas documentation for 'query' and 'eval'. If you want to add another criterion to the existing
        selection you can also do sth. like 'selected and hit == 1'.
        :param query: if left empty, defaults to 'True' -> selects all shots.
        :return:
        """
        selection = self._shots.eval(query)
        if selection.dtype != bool:
            raise ValueError('query must return a boolean!')
        self._shots.selected = selection
        print(f'{self._shots.selected.sum()} shots out of {self._shots.shape[0]} selected.')

    def change_filenames(self, file_suffix: Optional[str] = '.h5', file_prefix: str = '',
                         new_folder: Union[str, None] = None,
                         fn_map: Union[pd.DataFrame, None] = None,
                         keep_raw=True):
        """
        Change file names in all lists using some handy modifications. The old file names are copied to a "file_raw"
        column, if not already present (can be overriden with keep_raw).
        :param file_suffix: add suffix to file, INCLUDING file extension, e.g. '_modified.h5'
        :param file_prefix: add prefix to actual filenames (not folder/full path!), e.g. 'aggregated_
        :param new_folder: if not None, changes folder name to this path
        :param fn_map: if not None, gives an explicit table (pd.DataFrame) with columns 'file' and 'file_new'
            that manually maps old to new filenames. All other parameters are ignored, if provided
        :param keep_raw: if True (default), does not change the file_raw column in the shot list,
            unless there is none yet (in which case the old file names are _always_ copied to keep_raw)
        :return: DataFrame with a map from old to new file names (for reference)
        """

        if fn_map is None:
            # name mangling pt. 1: make map of old names to new names
            fn_map = self._shots[['file']].drop_duplicates()
            folder_file = fn_map.file.str.rsplit('/', 1, expand=True)
            if file_suffix is not None:
                new_fn = file_prefix + folder_file[1].str.rsplit('.', 1, expand=True)[0] + file_suffix
            else:
                new_fn = file_prefix + folder_file[1]
            if new_folder is not None:
                new_fn = new_folder + '/' + new_fn
            else:
                new_fn = folder_file[0] + '/' + new_fn
            fn_map['file_new'] = new_fn
            # print(fn_map)

        if (fn_map['file'] == fn_map['file_new']).all():
            warn('New and old file names are the same! Nothing will happen.')
            return fn_map

        # name mangling pt. 2: change names in all tables
        for lbl in ['_shots', '_peaks', '_predict', '_features']:
            table = self.__dict__[lbl]
            if table.shape[0] == 0:
                continue
            newtable = table.merge(fn_map, on='file', how='left').drop('file', axis=1). \
                rename(columns={'file_new': 'file'})
            if (lbl == '_shots') and (not keep_raw or 'file_raw' not in table.columns):
                newtable['file_raw'] = table.file

            self.__dict__[lbl] = newtable
            self.__dict__[lbl + '_changed'] = True

        # invalidate all the hdf file references (note that references into old files might still exist)
        self._file_handles = {}

        return fn_map

    def reset_id(self, keep_raw=True):
        """
        Resets shot_in_subset and Event columns to continuous numbering. Useful after dataset reduction. The old
        Event strings are copied to a "Event_raw" column, if not already present (can be overriden with keep_raw).
        :param keep_raw: if True (default), does not change the Event_raw column in the shot list,
            unless there is none yet (in which case the old Event IDs are _always_ copied to keep_raw)
        :return:
        """

        id_map = self._shots[self._shot_id_cols].copy()
        id_map['new_sis'] = self._shots.groupby(['file', 'subset']).cumcount()
        id_map['new_Event'] = self._shots.subset + '//' + id_map['new_sis'].astype(str)

        for lbl in ['_shots', '_peaks', '_predict']:
            table = self.__dict__[lbl]
            if table.shape[0] == 0:
                continue
            cols = {'new_Event': 'Event', 'new_sis': 'shot_in_subset'} if 'shot_in_subset' in table.columns \
                else {'new_Event': 'Event'}
            newtable = table.merge(id_map[self._shot_id_cols + list(cols.keys())], on=self._shot_id_cols, how='left'). \
                drop(list(cols.values()), axis=1).rename(columns=cols)
            if (lbl == '_shots') and (not keep_raw or 'Event_raw' not in table.columns):
                newtable['Event_raw'] = table.Event

            self.__dict__[lbl] = newtable
            self.__dict__[lbl + '_changed'] = True

    def init_files(self, overwrite=False, keep_features=False, exclude_list=()):
        """
        Make new files corresponding to the shot list, by copying over instrument metadata and maps (but not
        results, shot list, data arrays,...) from the raw files (as stored in file_raw).
        :param overwrite: overwrite new files if not yet existing
        :param keep_features: copy over the feature list from the files
        :param exclude_list: custom list of HDF5 groups or datasets to exclude from copying
        :return:
        """
        fn_map = self.shots[['file', 'file_raw']].drop_duplicates()

        exc = ('%/detector/data', self.data_pattern + '/%',
               self.result_pattern + '/%', self.shots_pattern + '/%')
        if not keep_features:
            exc += (self.map_pattern + '/features', '%/ref/features')
        if len(exclude_list) > 0:
            exc += tuple(exclude_list)

        # print(fn_map)

        if self.parallel_io:
            with ProcessPoolExecutor() as p:
                futures = []
                for _, filepair in fn_map.iterrows():
                    futures.append(p.submit(nexus.copy_h5,
                                            filepair['file_raw'], filepair['file'], mode='w' if overwrite else 'w-',
                                            exclude=exc,
                                            print_skipped=False))

                wait(futures, return_when=FIRST_EXCEPTION)
                for f in futures:
                    if f.exception():
                        raise f.exception()
                return futures

        else:
            for _, filepair in fn_map.iterrows():
                nexus.copy_h5(filepair['file_raw'], filepair['file'], mode='w' if overwrite else 'w-',
                              exclude=exc,
                              print_skipped=False)

            return None

    def get_meta(self, path='/%/instrument/detector/collection/shutter_time'):
        meta = {}    
        for lbl, _ in self.shots.groupby(['file', 'subset']):
            with h5py.File(lbl[0]) as fh:
                meta[tuple(lbl)] = fh[path.replace('%', lbl[1])][...]
                #print(type(meta[tuple(lbl)]))
                #print(meta[tuple(lbl)].shape)
                if meta[tuple(lbl)].ndim == 0:
                    meta[tuple(lbl)] = meta[tuple(lbl)][()]
                elif meta[tuple(lbl)].size == 1:
                    meta[tuple(lbl)] = meta[tuple(lbl)][0]
        return pd.Series(meta, name=path.rsplit('/',1)[-1])

    def merge_meta(self, path='%/instrument/detector/collection/shutter_time'):
        meta = self.get_meta(path)
        self.shots = self.shots.join(meta, on=['file', 'subset'])

    def get_selection(self, query: Union[str, None] = None,
                      file_suffix: Optional[str] = '_sel.h5', file_prefix: str = '',
                      new_folder: Union[str, None] = None,
                      reset_id: bool = True) -> 'Dataset':
        """
        Returns a new dataset object by applying a selection. By default, returns all shots with selected == True.
        Optionally, a different query string can be supplied (which leaves the selection unaffected).
        The stored file names will be changed, to avoid collisions. This can be controlled with the file_suffix and
        file_prefix parameters.
        :param query: Optional query string, as in the select method
        :param file_suffix: as in Dataset.change_filenames
        :param file_prefix: as in Dataset.change_filenames
        :param new_folder: as in Dataset.change_filenames
        :return: new data set.
        """

        if query is not None:
            cur_sel = self._shots.selected.copy()
            self.select(query)

        try:
            newset = copy.copy(self)
            newset._shots = self._sel().reset_index(drop=True)
            newset._peaks = self._sel(self._peaks).reset_index(drop=True)
            newset._predict = self._sel(self._predict).reset_index(drop=True)
            newset._features = self._features.merge(newset._shots[self._feature_id_cols],
                                                    on=self._feature_id_cols, how='inner', validate='1:m').\
                drop_duplicates(self._feature_id_cols).reset_index(drop=True)
            newset._stacks = {}

            newset.change_filenames(file_suffix, file_prefix, new_folder)
            if reset_id:
                newset.reset_id()
            newset._file_handles = {}

            if not self._stacks_open:
                warn('Getting selection, but stacks are not open -> not taking over stacks.')
            else:
                for k, v in self.stacks.items():
                    newset.add_stack(k, self._sel(v))
                # newset._stacks_open = False # this is a bit dodgy, but required to not have issues later...

        finally:
            if query is not None:
                self._shots.selected = cur_sel

        return newset

    def aggregate(self, by: Union[list, tuple] = ('sample', 'region', 'run', 'crystal_id'),
                  how: Union[dict, str] = 'mean',
                  file_suffix: str = '_agg.h5', file_prefix: str = '', new_folder: Union[str, None] = None,
                  query: Union[str, None] = None, force_commensurate: bool = True,
                  exclude_stacks: Optional[list] = None) -> 'Dataset':
        """
        Aggregate sub-sets of stacks using different aggregation functions. Typical application: sum sub-stacks of
        dose fractionation movies, or shots with different tilt angles (quasi-precession)
        :param by: shot table columns to group by for aggregation. Default: ('run', 'region', 'crystal_id', 'sample')
        :param how: aggregation types. 'sum', 'mean', and 'cumsum' are supported. Can be a single operation for all
            stacks, or a dict, specifying different ones for each, in which case all non-explicitly specified stacks
            will default to 'mean', e.g. how={'raw_counts': 'sum'}.
        :param file_suffix: as in Dataset.change_filenames
        :param file_prefix: as in Dataset.change_filenames
        :param new_folder: as in Dataset.change_filenames
        :param query: apply a query (see select or get_selection)
        :return: a new data set with aggregation applied
        """

        #TODO: fast agg only works on 3D arrays currently!
        # from time import time
        # T0 = time()
        by = list(by)
        newset = copy.copy(self)
        newset._stacks = {}
        exclude_stacks = [] if exclude_stacks is None else exclude_stacks

        if not self._stacks_open:
            raise RuntimeError('Stacks are not open.')
        
        # PART 1: MAKE A NEW SHOT TABLE ---
        
        # get shot selection and aggregation groups
        shsel = self.shots.reset_index(drop=True).query(query) if query is not None else \
            self.shots.reset_index(drop=True).groupby(by, sort=False)
        gb = shsel.groupby(by, sort=False)
        
        # get shot list columns that are (non-)identical within each aggregation group
        nonid = (gb.nunique() != 1).any()
        cols_nonid = list(nonid[nonid].index)
        cols_id = list(nonid[np.logical_not(nonid)].index)

        # add some information to shot list
        sh_initial = pd.concat([shsel, gb.ngroup().rename('_agg_grp_id'), gb.cumcount().rename('_agg_shot_in_grp')], axis=1)

        # re-sort initial table if required
        monotonous = (sh_initial['_agg_grp_id'][1:].values - sh_initial['_agg_grp_id'][:-1].values >=0).all()
        if not monotonous:
            sh_initial.sort_values(by=['_agg_grp_id','_agg_shot_in_grp'], inplace=True)
            
        # some sanity checks and status report
        by_frame = (sh_initial['frame'] - sh_initial['_agg_shot_in_grp']).nunique() == 1
        by_run = (sh_initial['run'] - sh_initial['_agg_shot_in_grp']).nunique() == 1
        print('Monotonous aggregation:', monotonous, '' if monotonous else '(PLEASE CHECK IF THIS IS DESIRED)')
        print('File/subset remixing:', ('file' in cols_nonid) or ('subset' in cols_nonid))
        print('Frame aggregation:', by_frame)
        print('Acq. run aggregation:', by_run)
        print('Discarding shot table columns:', cols_nonid)

        # generate mandatory cols (if files/subsets are remixed):
        def generate_common_name(name_list):
            from functools import reduce
            from difflib import SequenceMatcher
            return reduce(lambda s1, s2: ''.join([s1[m.a:m.a + m.size] for m in
                                                SequenceMatcher(None, s1, s2).get_matching_blocks()]), name_list)
        missing = [fn for fn in ['file', 'subset'] if fn not in cols_id]
        ssfields = gb[missing].aggregate(generate_common_name) if missing else None

        # compute final shot list
        sh_final = pd.concat([gb[cols_id + ['shot_in_subset', 'Event']].aggregate(lambda x: x.iloc[0]), gb.size().rename('agg_len'), ssfields], axis=1)
        newset._shots = sh_final.reset_index()
        
        # PART 2: DATA STACKS ---
            
        # aggregating functions
        func_lib = {'sum': lambda x: np.sum(x, axis=0, keepdims=True),
                    'mean': lambda x: np.mean(x, axis=0, keepdims=True),
                    'first': lambda x: x[:1,...],
                    'last': lambda x: x[-1:,...]}
            
        for sn, s in self.stacks.items():

            if sn in exclude_stacks:
                continue

            method = how.get(sn, 'first') if isinstance(how, dict) else how
            func = method if callable(method) else func_lib[method]

            # sliced and re-ordered stack
            stk_sel = s[sh_initial.index.values,...]

            # aggregated stack
            try:
                stk_agg = _map_sub_blocks(stk_sel, labels=sh_initial['_agg_grp_id'].values,
                                      func=func, dtype=s.dtype, name='aggregate_'+method, aggregating=True)
            except IndexError:
                raise ValueError(f'Unknown aggregation method {method}. Allowed ones are {tuple(func_lib.keys())}')
            except ValueError as e:
                if str(e).startswith('Mismatched chunk structure'):
                    warn(f'Stack '+sn+' has mismatched chunk structure. Rechunking to minimum chunk sizes. '
                         'Consider rechunking manually before, to improve performance.')
                    #TODO this comes with quite a performance penalty, but sth more complex would be comlex.
                    stk_rec = stk_sel.rechunk({0: tuple(sh_final['agg_len'].values)})
                    stk_agg = _map_sub_blocks(stk_rec,
                                              labels=sh_initial['_agg_grp_id'].values, 
                                              func=func, dtype=s.dtype, name='aggregate_'+method, 
                                              aggregating=True)
                else:
                    print('Error during aggregation of stack ' + sn)
                    raise e
            
            newset.add_stack(sn, stk_agg)

        # PART 3: OTHER STUFF ---

        try:
            newset._features = self._features.merge(newset._shots[self._feature_id_cols],
                                                    on=self._feature_id_cols, how='inner', validate='1:m'). \
                drop_duplicates(self._feature_id_cols).reset_index(drop=True)
        except Exception as e:
            warn('Could not aggregate features. Leaving them all in.')

        newset._file_handles = {}
        newset.change_filenames(file_suffix, file_prefix, new_folder, keep_raw=True)
        newset.reset_id(keep_raw=True)

        return newset
    
    def transform_stack_groups(self, stacks: Union[List[str], str], 
                               func: Callable[[np.ndarray], np.ndarray] = lambda x: np.cumsum(x, axis=0),
                               by: Union[List[str], Tuple[str]] = ('sample', 'region', 'run', 'crystal_id')):
        """For all data stacks listed in stacks, transforms sub-stacks within groups defined by "by".
        As a common example, applies some function to all frames of a diffraction movie. The dimensions of
        each sub-stack must not change in the process. Note that this happens in place, i.e., the stacks
        will be overwritten by a transformed version.
        A typical application is to calculate a cumulative sum of patterns wittin each diffraction movie. This
        is what the default parameter for func is doing. Can do all kinds of other fun things, i.e. calculating
        directly the difference between frames, the difference of each w.r.t. the first,
        normalizing them to sth, etc.

        Arguments:
            stacks {List or str} -- Name(s) of data stacks to be transformed

        Keyword Arguments:
            func {Callable} -- Function applied to each sub-stack. Must act on a numpy
                array and return one of the same dimensions. 
                Defaults to: lambda x: np.cumsum(x, axis=0)
            by {List} -- Shot table columns to identify groups. 
                Defaults to ['sample', 'region', 'run', 'crystal_id']
        """
        
        stacks = [stacks] if isinstance(stacks, str) else stacks
        by = list(by)
        feature_id = self.shots.groupby(by, sort=False).ngroup().values
        for sn in stacks:
            transformed = _map_sub_blocks(self.stacks[sn], feature_id, func, aggregating=False)
            self.add_stack(sn, transformed, overwrite=True)

    def init_stacks(self):
        """
        Opens stacks briefly, to check their sizes etc., and closes them again right away. Helpful to just get their
        names and sizes...
        :return:
        """
        self.open_stacks(init=True, readonly=True)
        self.close_stacks()

    def close_stacks(self):
        """
        Close the stacks by closing the HDF5 data file handles.
        :return:
        """
        for f in self._file_handles.values():
            f.close()
        self._file_handles = {}
        self._stacks_open = False

    def open_stacks(self, labels: Union[None, list] = None, checklen=True, init=False, 
        readonly=True, swmr=False, chunking: Union[int, str, list, tuple] = 'dataset'):
        """
        Opens data stacks from HDF5 (NeXus) files (found by the "data_pattern" attribute), and assigns dask array
        objects to them. After opening, the arrays or parts of them can be accessed through the stacks attribute,
        or directly using a dataset.stack syntax, and loaded using the .compute() method of the arrays.
        Note that, while the stacks are open, no other python kernel will be able to access the data files, and other
        codes may not see the modifications made. You will have to call dataset.close_stacks() before.
        :param labels: list of stacks to open. Default: None -> opens all stacks
        :param checklen: check if stack heights (first dimension) is equal to shot list length
        :param init: do not load stacks, just make empty dask arrays. Usually the init_stacks method is more useful.
        :param readonly: open HDF5 files in read-only mode
        :param swmr: open HDF5 files in SWMR mode
        :param chunking: how should the dask arrays be chunked along the 0th (stack) direction. Options are: 
            * an integer number for a defined (approximate) chunk size, which ignores shots with frame number < -1,
            * 'dataset' to use the chunksize of the dataset, 
            * an iterable to explicitly, and 
            * 'auto' to use the dask automatic. 
            * 'existing' to use the chunking of an already-existing stack which is about to be overwritten
            Generally, a fixed number (integer or iterable) is recommended and gives the least trouble.
            already been chunked before. For a fixed number, the chunks are done such that after filtering of frame == -1 shots,
            a constant chunk size is achieved.
        :return:
        """
        # TODO offer even more sophisticated chunking which always aligns with frames
        if 'frame' in self._shots.columns:
            sets = self._shots[['file', 'subset', 'shot_in_subset', 'frame']].drop_duplicates() # TODO why is the drop duplicates required?
        else:
            sets = self._shots[['file', 'subset', 'shot_in_subset']].drop_duplicates()
            sets['frame'] = 0
        stacks = defaultdict(list)

        if isinstance(chunking, (list, tuple)):
            chunking = list(chunking)[::-1]

        for (fn, subset), subgrp in sets.groupby(['file', 'subset']):
            self._file_handles[fn] = fh = h5py.File(fn, swmr=swmr, mode='r' if readonly else 'a')

            if isinstance(chunking, int) and (subgrp.frame == -1).any():
                # print('Found auxiliary frames, adjusting chunking...')
                # frames = subgrp[['frame']].copy()
                blocks = ((subgrp['frame'] != -1).astype(int).cumsum()-1) // chunking
                zchunks = tuple(subgrp.groupby(blocks)['frame'].count())
            elif isinstance(chunking, int):
                zchunks = chunking
            elif isinstance(chunking, list):
                chk = []
                Nshot = len(subgrp)
                while Nshot > 0:
                    chk.append(chunking.pop())
                    Nshot -= chk[-1]
                    if Nshot < 0:
                        raise ValueError('Requested chunking is incommensurate with file/subset boundaries!')
                zchunks = tuple(chk)
                # print(f'{fn}:{subset} -> {len(zchunks)} chunks.')
            else:
                zchunks = None

            grp = fh[self.data_pattern.replace('%', subset)]
            if isinstance(grp, h5py.Group):
                for dsname, ds in grp.items():
                    if ds is None:
                        # can happen for dangling soft links
                        continue
                    if ((labels is None) or (dsname in labels)) \
                            and isinstance(ds, h5py.Dataset) \
                            and ('pandas_type' not in ds.attrs):
                        # h5 dataset for file/subset found!
                        if checklen and (ds.shape[0] != subgrp.shape[0]):
                            raise ValueError(f'Stack height mismatch in f{fn}:{subset}. ' +
                                             f'Expected {subgrp.shape[0]} shots, found {ds.shape[0]}.')
                        
                        if zchunks is not None:
                            chunks = (zchunks,) + ds.chunks[1:]
                        elif chunking == 'dataset':
                            chunks = ds.chunks
                        elif chunking == 'auto':
                            chunks = 'auto'
                        elif chunking == 'existing':
                            chunks = self._stacks[dsname][subgrp.index.values,...].chunks
                        else:
                            raise ValueError('chunking must be an integer, list, tuple, "dataset", or "auto".')
                        
                        stackname = '_'.join([os.path.basename(fn).rsplit('.', 1)[0],
                                                subset, dsname])

                        if init:
                            newstack = da.empty(shape=ds.shape, dtype=ds.dtype, chunks=chunks, name=stackname)
                        else:
                            # print('adding stack: '+ds.name)
                            newstack = da.from_array(ds, chunks=chunks, name=stackname)
                        stacks[dsname].append(newstack)

        for sn, s in stacks.items():
            try:
                self._stacks[sn] = da.concatenate(s, axis=0) 
            except ValueError:
                warn(f'Could not read stack {sn}')

        self._stacks_open = True

    @contextmanager
    def Stacks(self, **kwargs):
        """Context manager to handle the opening and closing of stacks.
        returns the opened data stacks, which are automatically closed
        once the context is left. Arguments are passed to open_stacks Example:
            with ds.Stacks(readonly=True, chunking='dataset') as stk:
                center = stk.beam_center.compute()
            print('Have', center.shape[0], 'centers.')
        """
        warn('Use of the Stacks context manager is deprecated and may cause pain and sorrow.', DeprecationWarning)
        self.open_stacks(**kwargs)
        yield self.stacks
        self.close_stacks()

    def add_stack(self, label: str, stack: Union[da.Array, np.array, h5py.Dataset], overwrite=False):
        """
        Adds a data stack to the data set
        :param label: label of the new stack
        :param stack: new stack, can be anything array-like
        :param overwrite: allows overwriting an existing stack
        :return:
        """

        if not self._stacks_open:
            raise RuntimeError('Please open stacks before adding another one.')

        if stack.shape[0] != self.shots.shape[0]:
            raise ValueError('Stack height must equal that of the shot list.')

        if (label in self._stacks.keys()) and not overwrite:
            raise ValueError(f'Stack with name {label} already exists. Set overwrite = True.')

        if not isinstance(stack, da.Array):
            ch = stack.ndim * [-1]
            ch[0] = self.zchunks
            stack = da.from_array(stack, chunks=tuple(ch))

        self._stacks[label] = stack

    def delete_stack(self, label: str, from_files: bool = False):
        """
        Delete a data stack
        :param label: stack label
        :param from_files: if True, the stack is also deleted in the HDF5 files. Default False.
        :return:
        """

        if not self._stacks_open:
            raise RuntimeError('Please open stacks before deleting.')

        try:
            del self._stacks[label]
        except KeyError:
            warn(f'Stack {label} does not exist, not deleting anything.')

        if from_files:
            for _, address in self._shots[['file', 'subset']].iterrows():
                path = self.data_pattern.replace('%', address.subset) + '/' + label
                #print(f'Deleting dataset {path}')
                try:
                    del self._file_handles[address['file']][path]
                except KeyError:
                    pass
                    #print(address['file'], path, 'not found!')
                    
    def persist_stacks(self, labels: Union[None, str, list] = None, exclude: Union[None, str, list] = None,
                       include_3d: bool = False):
        
        if labels is None:
            labels = list(self._stacks.keys())
        elif isinstance(labels, str):
            labels = [labels]          
              
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
            
        if not include_3d:
            exclude.extend([sn for sn, stk in self._stacks.items() if stk.ndim >= 3])
        
        labels = [l for l in labels if l not in exclude]
        
        print('Persisting stacks to memory:', ', '.join(labels))
        self._stacks.update(dask.persist({sn: stk for sn, stk in self.stacks.items() if stk.ndim < 3})[0])

    def store_stacks(self, labels: Union[None, str, list] = None, exclude: Union[None, str, list] = None, overwrite: bool = False, 
                    compression: Union[str, int] = 32004, lazy: bool = False, data_pattern: Union[None,str] = None, 
                    progress_bar=True, chunking: Union[str, int] = 1, scheduler: str = 'threading', **kwargs):
        """
        Stores stacks with given labels to the HDF5 data files. If None (default), stores all stacks. New stacks are
        typically not yet computed, so at this point the actual data crunching is done. Note that this way of computing
        and storing data is restricted to threading or single-threaded computation, i.e. it's not recommended for
        heavy lifting. In this case, better use store_stack_fast.
        :param labels: stack(s) to be written
        :param exclude: stack(s) to be excluded
        :param overwrite: overwrite stacks already existing in the files?
        :param compression: compression algorithm to be used. 32004 corresponds to bz4, which we mostly use.
        :param lazy: if True, instead of writing the shots, returns two lists containing the arrays and dataset objects
                        which can be used to later pass them to dask.array.store. Default False (store right away)
        :param data_pattern: store stacks to this data path (% is replaced by subset) instead of standard path.
                        Note that stacks stored this way will not be retrievable through Dataset objects.
        :param progress_bar: show a progress bar during calculation/storing. Disable if you're running store_stacks
                        in multiple processes simultaneously.
        :param chunking: 0-dimension (stacking) chunk size. Can be fixed or 'stacks', to use the chunks of the dataset's. 
                        dask arrays holding the stacks (dataset.zchunks[0] - note that only the size of the _first_ chunk 
                        is used for all chunks.)
                        Defaults to 1, which is highly recommended.
        :param scheduler: dask scheduler to be used. Can be 'threading' or 'single-threaded'
        :param **kwargs: will be forwarded to h5py.create_dataset
        :return:
        """

        if not self._files_writable:
            raise RuntimeError('Please open files in write mode before storing.')

        if labels is None:
            labels = self._stacks.keys()
        elif isinstance(labels, str):
            labels = [labels]

        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
            
        labels = [l for l in labels is l not in exclude]
            
        cs = self.zchunks[0] if isinstance(chunking, str) and chunking.startswith('stack') else chunking

        stacks = {k: v for k, v in self._stacks.items() if k in labels}
        stacks.update({'index': da.from_array(self.shots.index.values, chunks=(cs,))})

        datasets = []
        arrays = []

        shots = self._shots.reset_index()  # just to be safe

        for (fn, ssn), sss in shots.groupby(['file', 'subset']):
            fh = self.file_handles[fn]

            if not all(np.diff(sss.shot_in_subset) == 1):
                raise ValueError(f'Non-continuous shot_in_subset in {fn}: {ssn}. Please sort out this mess.')

            if all(np.diff(sss.index) == 1):
                stack_idcs = slice(sss.index[0], sss.index[-1] + 1)
            else:
                stack_idcs = sss.index.values  # pathological case: non-continuous region in shot list
                warn(f'Shots for {fn}: {ssn} are non-contiguous in the shot list. Might hint at trouble.')

            for label, stack in stacks.items():
                # print(label)
                arr = stack[stack_idcs, ...]
                if data_pattern is None:
                    path = self.data_pattern.replace('%', ssn) + '/' + label
                else:
                    path = data_pattern.replace('%', ssn) + '/' + label

                #print('Writing to ', path)
                try:

                    # print(path, cs, arr.shape)
                    ds = fh.create_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                           chunks=(cs,) + arr.shape[1:],
                                           compression=compression, **kwargs)
                except RuntimeError as e:
                    if overwrite or label == 'index':
                        ds = fh.require_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                                chunks=(cs,) + arr.shape[1:],
                                                compression=compression, **kwargs)
                    else:
                        print('Cannot write stack', label)
                        raise e

                arrays.append(arr)
                datasets.append(ds)

        if lazy:
            return arrays, datasets

        else:
            with catch_warnings():
                if progress_bar:
                    with ProgressBar():
                        da.store(arrays, datasets, scheduler=scheduler)
                else:
                    da.store(arrays, datasets, scheduler=scheduler)

            for fh in self.file_handles.values():
                fh.flush()
                
    def store_stack_fast(self, label: str, client: Optional[Client] = None, sync: bool = True,
                         compression: Union[int, str] = 32004):
        """Store (and compute) a single stack to HDF5 file(s), using a dask.distributed cluster.
        This allows for proper parallel computation (even on many machines) and is wa(aaa)y faster
        than the standard store_stacks, which only works with threads.

        Arguments:
            label {str} -- Label of the stack to be computed and stored

        Keyword Arguments:
            client {Optional[Client]} -- dask.distributed client object. Mandatorily required 
                if sync=True. (default: {None})
            sync {bool} -- if True (default), computes and stores immediately, and returns a pandas 
                dataframe containing metadata of everything stored, for validation. If False,
                returns a list of dask.delayed objects which encapsulate the computation/storage.
            compression {Union[int, str]} -- Compression of the dataset to be stored. 
                Defaults to 32004, which is LZ4. Viable alternatives are 'gzip', 'lzf', or 'none'.

        Raises:
            ValueError: [description]

        Returns:
            pd.DataFrame (if sync=True), list of dask.delayed (if sync=False)
            
        Remarks:
            If the stack to be stored depends on computationally heavy (but memory-fitting) dask
            arrays which you want to retain outside this computation (e.g. to store them using
            store_stacks), consider persisting them (using da.persist) before calling sthis function.
            Otherwise, they will be re-calculated from scratch.
        """

        if self._files_open and not self._files_writable:
            raise RuntimeError('Please open files in write mode or close them before storing.')
        
        from distributed import Lock

        stack = self._stacks[label]
            
        # initialize datasets in files
        for (file, subset), grp in self.shots.groupby(['file', 'subset']):
            with h5py.File(file) as fh:
                fh.require_dataset(f'/{subset}/data/{label}', 
                                        shape=(len(grp),) + stack.shape[1:], 
                                        dtype=stack.dtype, 
                                        chunks=(1,) + stack.shape[1:], 
                                        compression=compression)
        
        self.close_stacks()
        
        chunk_label = np.concatenate([np.repeat(ii, cs) for ii, cs in enumerate(stack.chunks[0])])
        stk_del = stack.to_delayed().squeeze()

        locks = {fn: Lock() for fn in self.files}

        dels = []
        for chk, (cl, sht) in zip(stk_del,self.shots.groupby(chunk_label)):
            assert len(sht.drop_duplicates(['file','subset'])) == 1
            ii_to = sht.shot_in_subset.values
            dels.append(dask.delayed(nexus._save_single_chunk)(chk, file=sht.file.values[0], subset=sht.subset.values[0], 
                                label=label, idcs=ii_to, data_pattern=self.data_pattern, 
                                lock=locks[sht.file.values[0]]))
            
        if not sync:
            return dels

        else:
            # THIS DOES THE ACTUAL COMPUTATION/DATA STORAGE
            if client is None:
                raise ValueError('If immediate computation is desired (sync=True), you have to provide a cluster.')
            import random
            random.shuffle(dels) # shuffling tasks to minimize concurrent file access
            chunk_info = client.compute(dels, sync=True)
            return pd.DataFrame(chunk_info, columns=['file', 'subset', 'path', 'shot_in_subset'])
        
    def compute_and_save(self, diff_stack_label: Optional[str] = None, list_file: Optional[str] = None, client: Optional[Client] = None, 
                         exclude_stacks: Union[str,List[str]] = None, overwrite: bool = False, persist_diff: bool = True, 
                         persist_all: bool = False, compression: Union[str, int] = 32004):
        """
        Compound method to fully compute a dataset and write it to disk. It is designed for completely writing HDF5 
        files from scratch, not to append to existing ones. Internally calls init_files, store_tables, store_stacks,
        store_stack_fast, and write_list. 

        Keyword Arguments:
            diff_stack_label {Optional[str]} -- [description] (default: {None})
            list_file {Optional[str]} -- [description] (default: {None})
            client {Optional[Client]} -- [description] (default: {None})
            exclude_stacks {Optional[List[str]]} -- [description] (default: {None})
            overwrite {bool} -- [description] (default: {False})
            persist_diff {bool} -- [description] (default: {True})
            persist_all {bool} -- [description] (default: {False})
            compression {Union[str, int]} -- [description] (default: {32004})

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
        #TODO generalize to storing several diffraction stacks using sync=False.
         
        if (diff_stack_label is not None) and (diff_stack_label not in self._stacks):
            raise ValueError(f'Stack {diff_stack_label} not found in dataset.')

        if (diff_stack_label is not None) and (client is None):
            raise ValueError(f'If a diffraction data stack is specified, you must supply a dask.distributed client object.')        
            
        for dn in {os.path.dirname(f) for f in self.files}:
            if dn:
                os.makedirs(dn, exist_ok=True)

        exclude_stacks = [exclude_stacks] if isinstance(exclude_stacks, str) else exclude_stacks
        exclude_stacks = [diff_stack_label] if exclude_stacks is None else [diff_stack_label] + exclude_stacks

        self.close_stacks()

        print('Initializing data files...')
        self.init_files(overwrite=overwrite)

        print('Storing meta tables...')
        self.store_tables()

        # store all data stacks except for the actual diffraction data
        self.open_stacks(readonly=False)
        
        meta_stacks = [k for k in self.stacks.keys() if k not in exclude_stacks]
        print(f'Storing meta stacks {", ".join(meta_stacks)}')
        with dask.config.set(scheduler='threading'):
            self.store_stacks(labels=meta_stacks, compression=compression, overwrite=overwrite)

        print(f'Storing diffraction data stack {diff_stack_label}... monitor progress at {client.dashboard_link} (or forward port if remote)')
        chunk_info = self.store_stack_fast('corrected', client, compression=compression)

        # make sure that the calculation went consistent with the data set
        for (sh, sh_grp), (ch, ch_grp) in zip(self.shots.groupby(['file', 'subset']), chunk_info.groupby(['file', 'subset'])):
            if any(sh_grp.shot_in_subset.values != np.sort(np.concatenate(ch_grp.shot_in_subset.values))):
                raise ValueError(f'Incosistency between calculated data and shot list in {sh[0]}: {sh[1]} found. Please investigate.')
                
        if list_file is not None:
            self.write_list(list_file)
            
        if persist_all:
            self.open_stacks(readonly=True, chunking='existing')
            
        elif persist_diff:
            self.open_stacks(labels=[diff_stack_label], readonly=True, chunking='existing')
            
    #     else:
    #         ds_compute.open_stacks(labels=[]) # only populate the file handle list

    def rechunk_stacks(self, chunk_height: int):
        c = chunk_height
        ss_chunk = self.shots.groupby(['file', 'subset']).size().apply(lambda l: ((l // c) * [c]) + [l % c])
        zchunks = np.concatenate([np.array(v) for v in ss_chunk])
        assert zchunks.sum() == self.shots.shape[0]
        for sn, s in self.stacks.items():
            self.add_stack(sn, s.rechunk({0: tuple(zchunks)}), overwrite=True)

    def stacks_to_shots(self, stack_labels: Union[str, list], shot_labels: Optional[Union[str, list]] = None):
        if isinstance(stack_labels, str):
            stack_labels = [stack_labels,]
        if shot_labels is None:
            shot_labels = stack_labels
        elif isinstance(shot_labels, str):
            shot_labels = [shot_labels,]
        with self.Stacks() as stk:
            for lbl_from, lbl_to in zip(stack_labels, shot_labels):
                if lbl_from not in stk:
                    warn(f'{lbl_from} not in stacks, skipping.')
                self.shots[lbl_to] = stk[lbl_from]
            
    def merge_pattern_info(self, ds_from: 'Dataset', merge_cols: Optional[List[str]] = None, 
                           by: Union[List[str], Tuple[str]] = ('sample', 'region', 'run', 'crystal_id')):
        """Merge shot table columns and peak data from another data set into this one, based
        on matching of the shot table columns specified in "by".

        Arguments:
            ds_from {Dataset} -- Diffractem Dataset to take information from

        Keyword Arguments:
            merge_cols {List} -- Shot table columns to take over from other data set. If None (default),
                all columns are taken over which are not present in the shot table currently
            by {List} -- Shot table columns to match by. 
                Defaults to ['sample', 'region', 'run', 'crystal_id']

        Raises:
            ValueError: [description] d
        """
        by = list(by)
        
        merge_cols = ds_from.shots.columns.difference(list(self.shots.columns) + 
                                                    ['_Event', '_file', 'file_event_hash']) \
                                                        if merge_cols is None else merge_cols
        
        sh_from = ds_from.shots.copy() # avoid side effects on ds_from
        sh_from['ii_from'] = range(len(sh_from))
        sel_shots = self.shots.merge(sh_from[by + list(merge_cols) + ['ii_from']], on=by, 
                                        how='left', validate='m:1', indicator=True)
        
        if not all(sel_shots._merge == 'both'):
            raise ValueError('Not all features present in the dataset are present in ds_from.')

        self.shots = sel_shots.drop('_merge', axis=1)

        peakdata = {k: v for k, v in ds_from._stacks.items() if (k.startswith('peak') or k=='nPeaks')}

        if not all(self.shots.ii_from.diff().fillna(1) == 1):
            peakdata = {k: v[self.shots.ii_from.values,...] for k, v in peakdata.items()}

        for k, v in peakdata.items():
            self.add_stack(k, v, overwrite=True)

    def merge_acquisition_data(self, fields: dict):
        # mange instrument (acquisition) data like exposure time etc. into shot list
        raise NotImplementedError('merge_acquisition_data not yet implemented')

    def write_list(self, listfile: str):
        """
        Writes the files in the dataset into a list file, containing each file on a line.
        :param listfile: list file name
        :return:
        """
        #TODO allow to export CrystFEL-style single-pattern lists
        with open(listfile, 'w') as fh:
            fh.write('\n'.join(self.files))

    def generate_virtual_file(self, filename: str, diff_stack_label: str,
                              kind: str = 'fake', virtual_size: int = 1024):
        """Generate a virtual HDF5 file containing the meta data of the dataset, but not the actual
        diffraction. Instead of the diffraction stack, either a dummy stack containing a constant only,
        or a virtual dataset with external links to the actual data files is created. While the former
        is useful for indexing using CrystFEL, the latter can serve to generate a file for quick preview.

        Arguments:
            filename {str} -- [description]

        Keyword Arguments:
            kind {str} -- [description] (default: {'fake'})
            virtual_size {int} -- [description] (default: {1024})

        Raises:
            NotImplementedError: [description]
        """
        #TODO maybe rather split the preview file functionality off. It's too different.
        raise NotImplementedError('Fix this')