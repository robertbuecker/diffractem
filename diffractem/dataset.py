# dedicated to the penguin
import pandas as pd
import numpy as np
import dask.array.gufunc
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from . import io, nexus, tools
from .stream_parser import StreamParser
# from .map_image import MapImage
import h5py
from typing import Union, Dict, Optional, List, Tuple, Callable
import copy
from collections import defaultdict
from warnings import warn, catch_warnings, simplefilter
from tables import NaturalNameWarning
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
from contextlib import contextmanager
import os

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
        self.data_pattern: str = '/%/data' 
        '''Path to data stacks in HDF5 files. % can be used as placeholder (as in CrystFEL). Default /%/data'''
        self.shots_pattern: str = '/%/shots'
        '''Path to shot table data in HDF5 files. % can be used as placeholder (as in CrystFEL). Default /%/shots'''
        self._fallback_shots_pattern: str = '/%/data/shots'
        self.result_pattern: str = '/%/results'
        '''Path to result data (peaks, predictions) in HDF5 files. % can be used as placeholder (as in CrystFEL). 
        Default /%/results. **Note that storing results in this way is discouraged and deprecated.**'''
        self.map_pattern: str = '/%/map'
        '''Path to map and feature data in HDF5 files. % can be used as placeholder (as in CrystFEL). Default /%/map'''
        self.instrument_pattern: str = '/%/instrument'
        '''Path to instrument metadat in HDF5 files. % can be used as placeholder (as in CrystFEL). Default /%/instrument'''
        self.parallel_io: bool = True
        '''Toggles if parallel I/O is attempted for datasets spanning many files. Note that this is independent
        from `dask.distributed`-based parallelization as in `store_stack_fast`. Default True, which is overriden
        if the Dataset comprises a single file only.'''

        # internal stuff
        self._file_handles = {}
        self._stacks = {}
        self._shot_id_cols = ['file', 'Event']
        self._feature_id_cols = ['crystal_id', 'region', 'sample']
        self._diff_stack_label = ''

        # tables: accessed via properties!
        self._shots = pd.DataFrame(columns=self._shot_id_cols + self._feature_id_cols + ['selected'])
        self._peaks = pd.DataFrame(columns=self._shot_id_cols)
        self._predict = pd.DataFrame(columns=self._shot_id_cols)
        self._features = pd.DataFrame(columns=self._feature_id_cols)

    def __str__(self):
        return (f'diffractem Dataset object spanning {len(self._shots.file.unique())} NeXus/HDF5 files\n-----\n'
                f'{self._shots.shape[0]} shots ({self._shots.selected.sum()} selected)\n'
                f'{self._peaks.shape[0]} peaks, {self._predict.shape[0]} predictions, '
                f'{self._features.shape[0]} features\n'
                f'{len(self._stacks)} data stacks: {", ".join(self._stacks.keys())}\n'
                f'Diffraction data stack: {self._diff_stack_label}\n'
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

    def __len__(self):
        return self.shots.shape[0]

    @property
    def _files_open(self) -> bool:
        """True if HDF5 files are open"""
        return all([isinstance(f, h5py.File) for f in self.file_handles.values()])
    
    @property
    def _files_writable(self) -> bool:
        """True if HDF5 files are open in write mode"""
        return self._files_open and all([f.mode != 'r' for f in self.file_handles.values()])
    
    @property
    def _stack_in_memory(self) -> dict:
        """For each stack, indicates whether the dask array is persisted in memory. This is done
        by comparing the task number to the chunk number, which might be inaccurate in 
        pathological cases"""
        return {sn: len(stk.dask) == np.product(stk.numblocks) for sn, stk in self._stacks.items()}

    @property
    def file_handles(self) -> dict:
        """Handles to the HDF5 files as a dict with keys matching the file name, if files are open.
        Otherwise returns None (for each file)."""
        return {fn: self._file_handles[fn] if fn in self._file_handles else None for fn in self.files}

    @property
    def stacks(self) -> dict:
        """Dictionary of data stacks of the Dataset."""
        return self._stacks

    @property
    def files(self) -> list:
        """List of HDF5 files which the Dataset is based on. Note that these files do not have
        to actually exist; but they will be written if any of the writing functions is called.
        Change the file names and directories using `change_filenames`, or direct editing of the
        shot table (*discouraged*)"""
        return list(self._shots['file'].unique())

    @property
    def shots(self) -> pd.DataFrame:
        """Shot list. Can be overwritten only if index and ID columns of the shots
        are identical to the existing one."""
        return self._shots

    @shots.setter
    def shots(self, value: pd.DataFrame):
        if (value.index != self._shots.index).any():
            raise ValueError('Shot index is different from existing one.')
        if (value[self._shot_id_cols] != self._shots[self._shot_id_cols]).any().any():
            raise ValueError('Shot ID columns are different from existing ones.')
        self._shots = value
        self._shots_changed = True

    @property
    def predict(self) -> pd.DataFrame:
        """List of predictions. Deprecated. Please store predictions in StreamParser objects."""
        warn('The prediction table functionality will likely be removed.', DeprecationWarning)
        return self._predict

    @predict.setter
    def predict(self, value):
        warn('The prediction table functionality will likely be removed.', DeprecationWarning)
        self._predict = value
        self._predict_changed = True
        
    @property
    def features(self) -> pd.DataFrame:
        """List of features (that is e.g. crystals). Each feature can have one or many shots
        associated with it."""
        return self._features

    @features.setter
    def features(self, value):
        self._features = value
        self._features_changed = True
        
    @property
    def peaks(self) -> pd.DataFrame:
        """List of found diffraction peaks. Deprecated. Please store peaks in CXI-format
        stacks. Note that peak positions in this table must follow *CrystFEL* convention, that
        is, integer numbers specify the pixel *edges*, not centers. This is in contrast to
        CXI convention, where integer numbers correspond to pixel *centers*"""
        warn('The peak table functionality will likely be removed.', DeprecationWarning)
        return self._peaks
            
    @peaks.setter
    def peaks(self, value):
        warn('The peak table functionality will likely be removed.', DeprecationWarning)
        self._peaks = value
        self._peaks_changed = True
            
    @property
    def peak_data(self) -> Dict[str, da.Array]:
        """Stored Bragg reflection data in CXI format, if present. Otherwise raises error."""
        if all([sn in self.stacks for sn in ['nPeaks', 'peakXPosRaw', 'peakYPosRaw']]):
            pkdat = {k: v for k, v in self.stacks.items() if k in ['nPeaks', 'peakXPosRaw', 'peakYPosRaw']}
            if 'peakTotalIntensity' in self.stacks:
                pkdat['peakTotalIntensity'] = self.stacks['peakTotalIntensity']
            return pkdat
        else:
            raise ValueError('No peak data found in dataset.')
        
    @peak_data.setter
    def peak_data(self, v: Dict[str, Union[da.Array, np.ndarray]]):
        if all([sn in v for sn in ['nPeaks', 'peakXPosRaw', 'peakYPosRaw']]):
            for k, v in v.items():
                self.add_stack(k, v, overwrite=True)
        else:
            raise ValueError('Supplied peak data is incomplete.')
            
    @property
    def zchunks(self) -> tuple:
        """Chunks of dask arrays holding the stacks along their first (that is, stacked) axis."""
        # z chunks of dataset stacks
        allchk = [stk.chunks[0] for stk in self._stacks.values()]
        if allchk and all([chk == allchk[0] for chk in allchk]):
            return allchk[0]
        elif allchk:
            warn('Stacks have unequal chunking along first axis. This is undesirable.')
        else:            
            return None        
        
    @property
    def diff_stack_label(self):
        """Label of stack which holds the diffraction data."""
        return self._diff_stack_label
    
    @diff_stack_label.setter
    def diff_stack_label(self, value):
        if value in self.stacks:
            self._diff_stack_label = value
        else: 
            ValueError(f'{value} is not a stack.')
        
    @property
    def diff_data(self) -> da.Array:
        """Returns diffraction data stack (as identified by the diff_stack_label property"""
        return self.stacks[self.diff_stack_label]
    
    @classmethod
    def from_files(cls, files: Union[list, str, tuple], open_stacks: bool = True, chunking: Union[int, str] = 'hdf5', 
                   persist_meta: bool = True, init_stacks: bool = False, load_tables: bool = True, 
                   diff_stack_label: str = 'raw_counts', validate_files: bool = False, **kwargs):
        """Create a `Dataset` object from HDF5 file(s) stored on disk.
        
        There is some flexibility with regards to how to define the input files. You can specify them by
        
        * a .lst file name, which contains a simple list of H5 files (on separate lines). If the .lst file has CrystFEL-style
          event indicators in it, it will be loaded, and the events present in the list will be selected, the others not.
        * a glob pattern (like: 'data/*.h5')
        * a python iterable of files. 
        * a simple HDF5 file path
            
        In any case, the shot list and feature list are loaded to memory. Using the arguments you can specify what should
        happen to the stacks.
            
        Args:
            files (Union[list, str, tuple]): File specification as decsribed above.
            open_stacks (bool, optional): Open the data stacks. This means that open handles to the HDF5 (in readonly mode). 
                are kept within the `Dataset` object. Defaults to True.
            chunking (Union[int, str], optional):  See documentation of `open_stacks`. Defaults to 'hdf5', that is, look
                up in the HDF5 file for a recommendation value.
            persist_meta (bool, optional): Right away persists the data stacks, that is, loads the actual data into memory
                instead of just holding references to the HDF5 files. Diffraction data (identified by 3D stacks) is automatically
                excluded. Defaults to True.
            init_stacks (bool, optional): Initialize stacks, that is, briefly open the data stacks, check their lengths, and close
                the files again. Viable option if you need/want to set open_stacks=False for some reason. Defaults to False.
            load_tables (bool, optional): Also load peaks and prediction tables from the HDF5 files. Defaults to True (will likely
                be changed to False).
            diff_stack_label (str, optional): Label of the diffraction data stack. Defaults to 'raw_counts'.
            validate_files (bool, optional): Validate the HDF5 files (that is, check for required groups and datasets)
                before attempting to open them. Defaults to False.
            **kwargs: Dataset attributes to be set right away.

        Returns:
            Dataset: new Dataset object read from files
        """

        file_list = io.expand_files(files, scan_shots=True, validate=validate_files)
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
        if init_stacks and not open_stacks:
            self.init_stacks(chunking=chunking)
        if load_tables:
            self.load_tables(features=True, peaks=True, predict=True)          
        if open_stacks:
            self.open_stacks(chunking=chunking)
        if open_stacks and persist_meta:
            self.persist_stacks(exclude=diff_stack_label, include_3d=False)

        return self
    
    from_list = from_files # for compatibility

    #TODO What is this method doing here? Shouldn't it go into some tool module?
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

    def load_tables(self, shots: bool = False, features: bool = False, 
                    peaks: bool = False, predict: bool = False, files: bool = None):
        """Load pandas metadata tables from the HDF5 files. Set the argument for the table you want to load to True.

        Args:
            shots (bool, optional): Get shot table. Defaults to False.
            features (bool, optional): Get feature table. Defaults to False.
            peaks (bool, optional): Get peaks table. Defaults to False.
            predict (bool, optional): Get prediction table. Defaults to False.
            files (bool, optional): Only include sub selection of files - usually not a good idea.
                Uses all files of dataset if None. Defaults to None.
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
                warn('No shots found at ' + self.shots_pattern)

        if features:
            try:
                self._features = nexus.get_table(files, self.map_pattern + '/features', parallel=self.parallel_io)
                # print(len(self._features))
                self._features_changed = False
                
            except KeyError as kerr:
                print(f'No feature list in data set ({str(kerr)}). That\'s ok if it\'s a virtual or info file.')
                # raise err

            try:
                
                if 'sample' not in self._features.columns:
                    sdat = nexus.get_meta_fields(list(self._features.file.unique()),
                                                 ['/%/sample/name', '/%/sample/region_id', '/%/sample/run_id']). \
                        rename(columns={'name': 'sample', 'region_id': 'region', 'run_id': 'run'})
                    self._features = self._features.merge(sdat, how='left', on=['file', 'subset'])

                self._features.drop_duplicates(self._feature_id_cols, inplace=True) # for multi files with identical features
                self._features.drop(columns=[c for c in ['file', 'subset'] if c in self._features.columns], inplace=True)

            except Exception as err:
                print('Error processing ' + self.map_pattern + '/features')
                raise err

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
        """Stores the metadata tables (shots, features, peaks, predictions) into HDF5 files. 
        
        The location into which the tables will be stored is defined in the Dataset object's attributes. The format
        in which they are stored is, on the other hand, determined by the *format* argument. If set to 'tables',
        PyTables will be used to store the table in a native HDF5 table format, which however is somewhat uncommon
        and not recognized by CrystFEL. If set to 'nexus' (Default), each column of the table will be stored as
        a one-dimensional dataset.
        
        As a general recommendation, **always** use nexus format to store the shots and features. For peaks and 
        predictions, 'tables' is rather preferred, as it allows faster read/write and is more flexible with
        regards to column labels.
        
        For each of the tables,
        it can be automatically determined if they have changed and should be stored (however, this only works if 
        no inplace changes have been made. So don't rely on it too much.). If you want this, leave the
        argument at None. Otherwise explicitly specify True or False (strongly recommended).

        Args:
            shots (Union[None, bool], optional): Store shot table. Defaults to None.
            features (Union[None, bool], optional): Store feature table. Defaults to None.
            peaks (Union[None, bool], optional): Store peak table. Defaults to None.
            predict (Union[None, bool], optional): Store prediction table. Defaults to None.
            format (str, optional): Table storage format in HDF5 file, can be 'nexus' or 'tables'. Defaults to 'nexus'.
        """
        
        fs = []
        
        if self._files_open and not self._files_writable:
            # files are open in read-only, they need to be closed
            stacks_were_open = True
            self.close_files()
        else:
            stacks_were_open = False

        simplefilter('ignore', NaturalNameWarning)
        if (shots is None and self._shots_changed) or shots:
            # sh = self.shots.drop(['Event', 'shot_in_subset'], axis=1)
            # sh['id'] = sh[['sample', 'region', 'run', 'crystal_id']].apply(lambda x: '//'.join(x.astype(str)), axis=1)
            fs.extend(nexus.store_table(self.shots, self.shots_pattern, parallel=self.parallel_io, format=format))
            self._shots_changed = False

        if (features is None and self._features_changed) or features:
            fs.extend(nexus.store_table(self.features.merge(self.shots[self._feature_id_cols + ['file', 'subset']], 
                                                            on=self._feature_id_cols, validate='1:m'), 
                                        self.map_pattern + '/features', parallel=self.parallel_io,
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
        """Loads a `CrystFEL` stream file and merges it contents into the dataset.

        Args:
            streamfile (Union[StreamParser, str]): stream file name, or StreamParser object.
        """

        # ...it would be way more elegant, to just associate a StreamParser object, and merge the list in
        # accessors. But the merges can become pretty slow for large files, so we do it only here.
        
        warn('Dataset.merge_stream is deprecated. Please use StreamParser to work with indexing results', DeprecationWarning)

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

    def get_map(self, file, subset='entry'):
        # TODO: get a MapImage from stored data, with tables filled in from dataset
        raise NotImplementedError('does not work yet, sorry.')

    def _sel(self, obj: Union[None, pd.DataFrame, da.Array, np.array, h5py.Dataset, list, dict] = None):
        """
        General internal method to pick items that belong to shots with selected==True from many kinds of data types.
        
        * For DataFrames, it matches the selected items by the datasets ID columns (usually 'file' and 'Event',
          or 'crystal_id' and 'region')
        * For anything slicable (dask or numpy array), it picks elements along the first array dimension,
          assuming that the stack is ordered the same way as the shot list.
        * Also accepts lists or dicts of all such objects and returns a corresponding list or dict.
        
        Args:
            obj: DataFrame, numpy Array, dask Array, h5py Dataset, list, dict
        
        Returns: 
            subset of input object (typically as non-copied view!)
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
        
        Args:
            query (str): if left empty, defaults to 'True' -> selects all shots.
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
        """Change file names in all lists using some handy modifications. 
        
        The old file names are copied to a "file_raw" column, if not already present 
        (can be overriden with keep_raw).

        Args:
            file_suffix (Optional[str], optional): add suffix to file, INCLUDING file extension, e.g. '_modified.h5'. 
                Defaults to '.h5', i.e., no change is made except for the file extension being fixed to h5.
            file_prefix (str, optional): add prefix to actual filenames (not folder/full path!), e.g. 'aggregated_'. 
                Defaults to '', i.e., no prefix..
            new_folder (Union[str, None], optional): If not None, changes the file folders to this path. Defaults to None.
            fn_map (Union[pd.DataFrame, None], optional): if not None, expects an explicit table (pd.DataFrame) with columns 
                'file' and 'file_new'
                that manually maps old to new filenames. *All other parameters are ignored, if provided.* Defaults to None.
            keep_raw (bool, optional): If True (default), does not change the file_raw column in the shot list,
                unless there is none yet (in which case the old file names are *always* copied to keep_raw). Defaults to True.
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
            if 'file' in table.columns:
                newtable = table.merge(fn_map, on='file', how='left').drop('file', axis=1). \
                    rename(columns={'file_new': 'file'})
            else:
                newtable = table
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
        
        Args:
            keep_raw (bool, optional): if True (default), does not change the Event_raw column in the shot list,
                unless there is none yet (in which case the old Event IDs are *always* copied to keep_raw)
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
        """Initialize set of HDF5 files to store the Dataset.
        
        Makes new files corresponding to the shot list, by creating the files with the basic structure, and
        copying over instrument metadata and maps (but not shot list, data arrays,...) 
        from the raw files (as stored in file_raw).

        Args:
            overwrite (bool, optional): Overwrite files if existing already. Defaults to False.
            keep_features (bool, optional): Copy over the (full) feature list. Usually not required,
                as it will be later stored using store_stacks. Defaults to False.
            exclude_list (tuple, optional): Custom list of HDF5 groups or datasets to exclude
                from copying. Please consult documentation of `nexus.copy_h5` for help. Defaults to ().
        """
        
        fn_map = self.shots[['file', 'file_raw']].drop_duplicates()

        exc = ('%/detector/data', self.data_pattern + '/%',
               self.result_pattern + '/%', self.shots_pattern + '/%')
        if not keep_features:
            exc += (self.map_pattern + '/features', '%/ref/features')
        if len(exclude_list) > 0:
            exc += tuple(exclude_list)

        # print(fn_map)P

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

        else:
            for _, filepair in fn_map.iterrows():
                nexus.copy_h5(filepair['file_raw'], filepair['file'], mode='w' if overwrite else 'w-',
                              exclude=exc,
                              print_skipped=False)


    def get_meta(self, path: str = '/%/instrument/detector/collection/shutter_time') -> pd.Series:
        """Gets an instrument metadata field in NeXus format from the HDF5 files. 
        
        As those metadata are per-file and not per-shot, a series is returned which then can be joined into
        the dataset manually. If you want to have this done automatically, use `merge_meta` instead.

        Args:
            path (str, optional): Path to metadata to be grabbed. Can include CrystFEL-stype % placeholder. 
                Defaults to '/%/instrument/detector/collection/shutter_time'.

        Returns:
            pd.Series: pandas Series holding the metadata for each file.
        """
        
        meta = {}    
        for lbl, _ in self.shots.groupby(['file', 'subset']):
            with h5py.File(lbl[0], 'r') as fh:
                meta[tuple(lbl)] = fh[path.replace('%', lbl[1])][...]
                #print(type(meta[tuple(lbl)]))
                #print(meta[tuple(lbl)].shape)
                if meta[tuple(lbl)].ndim == 0:
                    meta[tuple(lbl)] = meta[tuple(lbl)][()]
                elif meta[tuple(lbl)].size == 1:
                    meta[tuple(lbl)] = meta[tuple(lbl)][0]
        return pd.Series(meta, name=path.rsplit('/',1)[-1])

    def merge_meta(self, path='%/instrument/detector/collection/shutter_time'):
        """Gets an instrument metadata field in NeXus format from the HDF5 files, and merges it into
        the shot table of the data set.
        
        Note, that the name of the new column in the shot table will correspond to the HDF5 dataset name,
        ignoring the group (as included in the full path). E.g., for the default value, it will be
        just 'shutter_time'.

        Args:
            path (str, optional): Path to metadata to be grabbed. Can include CrystFEL-style % placeholder. 
                Defaults to '%/instrument/detector/collection/shutter_time'.
        """
        meta = self.get_meta(path)
        self.shots = self.shots.join(meta, on=['file', 'subset'])

    def get_selection(self, query: Union[str, None] = None,
                      file_suffix: Optional[str] = '_sel.h5', file_prefix: str = '',
                      new_folder: Union[str, None] = None,
                      reset_id: bool = True) -> 'Dataset':
        """Returns a new dataset object by applying a selection. 
        
        By default, returns a new Dataset object, including all shots with selected == True in the current shot list.
        Optionally, a different query string can be supplied (which leaves the selection unaffected).
        The file names of the new data set will be changed, to avoid collisions. This can be controlled with the file_suffix and
        file_prefix parameters. Otherwise, the returned dataset will include everything from the existing one.
        
        Hint: 

        Args:
            query (Union[str, None], optional): Optional query string, as in the `select` method. Defaults to None, that is,
                use the `selected` column in the shot list.
            file_suffix (Optional[str], optional): as in `change_filenames`. Defaults to '_sel.h5'.
            file_prefix (str, optional): as in `change_filenames`. Defaults to ''.
            new_folder (Union[str, None], optional): as in `change_filenames`. Defaults to None.
            reset_id (bool, optional): reset the shot in subset. Defaults to True.

        Returns:
            Dataset: New dataset with all the same attributes, but containing only the desired sub-selection of shots.
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

            for k, v in self.stacks.items():
                newset.add_stack(k, self._sel(v))
                
            newset.persist_stacks([sn for sn, inmem in self._stack_in_memory.items() if inmem])

        finally:
            if query is not None:
                self._shots.selected = cur_sel

        return newset

    def copy(self, file_suffix: Optional[str] = '_copy.h5', 
                    file_prefix: str = '', 
                    new_folder: Union[str, None] = None) -> 'Dataset':
        """Makes a (deep) copy of a dataset, changing the file names.
        
        Internally, this just calls `get_selection` with `query='True'`.

        Args:
            file_suffix (Optional[str], optional): as in `change_filenames`. Defaults to '_copy.h5'.
            file_prefix (str, optional): as in `change_filenames`. Defaults to ''.
            new_folder (Union[str, None], optional): as in `change_filenames`. Defaults to None.

        Returns:
            Dataset: Copy of the dataset
        """
        
        return self.get_selection('True', file_suffix, file_prefix, new_folder)

    def aggregate(self, by: Union[list, tuple] = ('sample', 'region', 'run', 'crystal_id'),
                  how: Union[dict, str] = 'sum',
                  file_suffix: str = '_agg.h5', file_prefix: str = '', new_folder: Union[str, None] = None,
                  query: Union[str, None] = None,
                  exclude_stacks: Optional[list] = None) -> 'Dataset':
        """Aggregate sub-sets of stacks (like individual diffraction movies) using different aggregation functions.
        
        Each set of shots with identical values of the columns specified in `by` will be squashed into a single
        one, using aggregation functions applied to the stacks as described in `how`. These can be different for each of
        the stacks. Unlike for the stacks, inconsistent fields in the shot list within each group are simply killed.
        The function finally returns a new dataset containing the aggregated data, it leaves the existing set untouched.
        
        The typical application is to sum sub-stacks of dose fractionation movies, or shots with different tilt angles 
        (quasi-precession). If you're familiar with pandas a bit, it's sort of like a `DataSet.GroupBy(by).agg' operation.
        
        In most cases (well-ordered data sets), this function should just work. More pathological ones are not
        sufficiently tested, though some sanity checks and precautions are taken.
        
        As an example: setting how=['sample', 'region', 'run', 'crystal_id'] (which is the default) will aggregate over
        all shots taken in a single run, and if you set how='sum', the stacks will be added.
        
        Args:
            by (Union[list, tuple], optional): shot table columns to group by for aggregation. 
                Defaults to ('sample', 'region', 'run', 'crystal_id').
            how (Union[dict, str], optional): string specifying the aggregation method for stacks. Allowed
                values are 'mean', 'sum', 'first', 'last'. You can also specify a dict with different values
                for each stack, like {'raw_counts': 'sum', 'nPeaks': 'first'}. Defaults to 'sum'.
            file_suffix (str, optional): as in `change_filenames`. Defaults to '_agg.h5'.
            file_prefix (str, optional): as in `change_filenames`. Defaults to ''.
            new_folder (Union[str, None], optional): as in `change_filenames`. Defaults to None.
            query (Union[str, None], optional): additional query to sub-select data before aggregation (as in 
                `select` or `get_selection). E.g. query='frame >= 1 and frame < 5" would only aggregate frames
                1 to 4. Defaults to None.
            exclude_stacks (Optional[list], optional): Exclude stacks from the aggregated dataset. Defaults to None.

        Returns:
            Dataset: Dataset containing the aggregated data
        """

        #TODO: fast agg only works on 3D arrays currently!
        # from time import time
        # T0 = time()
        by = list(by)
        newset = copy.copy(self)
        newset._stacks = {}
        exclude_stacks = [] if exclude_stacks is None else exclude_stacks
        
        # PART 1: MAKE A NEW SHOT TABLE ---
        
        # get shot selection and aggregation groups
        shsel = self.shots.reset_index(drop=True).query(query) if query is not None else \
            self.shots.reset_index(drop=True)
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
                    warn(f'Stack {sn} has mismatched chunk structure. Rechunking to minimum chunk sizes. '
                         'Consider rechunking manually before, to improve performance.')
                    #TODO this comes with quite a performance penalty, but sth more complex would be complex.
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
        newset.persist_stacks([sn for sn, inmem in newset._stack_in_memory.items() if inmem])        

        return newset
    
    def transform_stack_groups(self, stacks: Union[List[str], str], 
                               func: Callable[[np.ndarray], np.ndarray] = lambda x: np.cumsum(x, axis=0),
                               by: Union[List[str], Tuple[str]] = ('sample', 'region', 'run', 'crystal_id')):
        """
        For all data stacks listed in stacks, transforms sub-stacks within groups defined by `by` using the
        function in `func`.
        
        The dimensions of each sub-stack must not change in the process. 
        Note that, unlike for `get_selection` or `aggregate`,
        this happens *in place*, i.e., the stacks will be **overwritten** by a transformed version! If this
        is not what you want, first make a copy of your data set, using `copy`.
        
        A typical application is to calculate a cumulative sum of patterns wittin each diffraction movie. This
        is what the default parameters for `by` and `func` is doing. Can do all kinds of other fun things, i.e. 
        calculating directly the difference between frames, the difference of each w.r.t. the first,
        normalizing them to sth, etc.

        Args:
            stacks: Name(s) of data stacks to be transformed
            func: Function applied to each sub-stack. Must act on a numpy
                array and return one of the same dimensions. Defaults to `lambda x: np.cumsum(x, axis=0)`.
            by: Shot table columns to identify groups - similar to how it's done in `aggregate`.
                Defaults to `('sample', 'region', 'run', 'crystal_id')`.

        """

        stacks = [stacks] if isinstance(stacks, str) else stacks
        by = list(by)
        feature_id = self.shots.groupby(by, sort=False).ngroup().values
        for sn in stacks:
            transformed = _map_sub_blocks(self.stacks[sn], feature_id, func, aggregating=False)
            self.add_stack(sn, transformed, overwrite=True)
        
        self.persist_stacks([sn for sn in stacks if self._stack_in_memory[sn]])

    def init_stacks(self, **kwargs):
        """Opens files briefly in readonly mode, to check stack names shapes etc., and closes them again right away. 

        Args:
            **kwargs: any arguments are passed to `open_stacks`
        
        """
        # warn('init_stacks is often not required. Double-check if you really need it.', DeprecationWarning)
        self.open_stacks(init=True, readonly=True, **kwargs)
        self.close_files()

    def close_files(self):
        """Closes all HDF5 files.
        
        Note that this might have side effects: if stacks are accessible that depend on non-persisted HDF5 datasets
        in the files, they will not be usable anymore after issuing this command and cause trouble especially
        for the distributed scheduler. So don't close the files unless you really have to.
        
        """
        for f in self._file_handles.values():
            f.close()
            del f
        self._file_handles = {}

    close_stacks = close_files

    def open_stacks(self, labels: Union[None, list] = None, checklen=True, init=False, 
        readonly=True, swmr=False, chunking: Union[int, str, list, tuple] = 'dataset'):
        """Opens data stacks from HDF5 (NeXus) files (found by the "data_pattern" attribute), and assigns dask array
        objects to them. After opening, the arrays or parts of them can be accessed through the stacks attribute,
        or directly using a dataset.stack syntax, and loaded using the .compute() or .persist() method of the arrays.
        
        A critical point here is how the chunking of the dask arrays is done. Especially for the initial opening
        of raw data this is crucial for (as in: orders of magnitude) the performance of downstream tasks. You have
        several options, those are, in decreasing order of recommendation:
        
        * 'dataset' to use what is set in the current dataset zchunks property (default). This will not work for a fresh
          dataset, in which case you have to specify it from scratch.
        * 'hdf5' to use the chunksize recommended in the HDF5 file ('recommended_zchunks' attribute) of the
          data stacks group.
        * an integer number for a defined (approximate) chunk size, which ignores shots with frame number < -1. This means,
          that after a get_selection command or anything that filters out dummy shots, equal chunk sizes are achieved.
          This is the recommended way of chunking for totally from-scratch datasets which don't yet have the
          recommended_zchunks attribute set. Something of the order of 10 is often a good choice if you want to work
          with the set as is, if you want to aggregate early on, choose something bigger (rather 100).
          **If your dataset comprises diffraction movies, this should be an integer
          multiple of the number of frames within each.**
        * an iterable to explicitly set the chunk sizes
        * 'existing' to use the chunking of an already-existing stack which is about to be overwritten.
          Should usually be the same as 'dataset', but still works if your stacks have inconsistent chunking.
        * 'auto' to use the dask automatic mode, with inevitably sub-optimal results.

        Args:
            labels (Union[None, list], optional): lLst of stacks to open. To open all stacks, set to None. Defaults to None.
            checklen (bool, optional): check if stack heights (first dimension) is equal to shot list length. Defaults to True.
            init (bool, optional): do not load stacks, just make empty dask arrays.  Defaults to False.
            readonly (bool, optional): open HDF5 files in read-only mode. Defaults to True.
            swmr (bool, optional): open HDF5 files in SWMR mode. Defaults to False.
            chunking (Union[int, str, list, tuple], optional): [description]. Defaults to 'dataset'.

        """
        #TODO DO NOT OVERWRITE PERSISTED STACKS!
        
        if (not readonly and self._files_open and not self._files_writable) or \
            (readonly and self._files_writable):
                
            if chunking != 'existing':
                warn('Reopening files in a different mode. Chunking will be set to "existing".')
                chunking = 'existing'
                
            # reopen the stacks in a different mode!
            self.close_files()
            
        if not readonly and self._files_writable and not swmr:
            # write access already. Nobody else had access anyway
            return
               
        # TODO offer even more sophisticated chunking which always aligns with frames
        if 'frame' in self._shots.columns:
            sets = self._shots[['file', 'subset', 'shot_in_subset', 'frame']].drop_duplicates() # TODO why is the drop duplicates required?
        else:
            sets = self._shots[['file', 'subset', 'shot_in_subset']].drop_duplicates()
            sets['frame'] = 0
        stacks = defaultdict(list)

        if isinstance(chunking, (list, tuple)):
            chunking = list(chunking)[::-1]
            
        elif(isinstance(chunking, str) and chunking=='dataset'):
            if self.zchunks is not None:
                chunking = list(self.zchunks)[::-1]
            else:
                raise ValueError('Dataset chunking is undefined (yet). You have to pick an explicit chunking option.')
            
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
            elif isinstance(chunking, str) and chunking == 'hdf5':
                try:
                    zchunks = tuple(fh[self.data_pattern.replace('%', subset)].attrs['recommended_zchunks'])
                except KeyError:
                    raise ValueError('The HDF5 files do\'nt have a chunking preset. Please specify chunking explicitly.')
            else:
                zchunks = None

            grp = fh[self.data_pattern.replace('%', subset)]
            if isinstance(grp, h5py.Group):
                try:
                    curr_lbl = grp.attrs['signal']
                    if not isinstance(curr_lbl, str):
                        curr_lbl = curr_lbl.decode()
                    if not self._diff_stack_label:
                        self._diff_stack_label = curr_lbl
                    elif self._diff_stack_label != curr_lbl:
                        warn(f'Non-matching primary diffraction stack labels: '
                            f'{self._diff_stack_label} vs {grp.attrs["signal"].decode()}')
                        
                except KeyError:
                    # no diff stack label stored
                    curr_lbl = self._diff_stack_label
                
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
                        elif chunking == 'hdf5':
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

    @contextmanager
    def Stacks(self, **kwargs):
        """Context manager to handle the opening and closing of stacks.
        returns the opened data stacks, which are automatically closed
        once the context is left. Arguments are passed to open_stacks Example:
            with ds.Stacks(readonly=True, chunking='dataset') as stk:
                center = stk.beam_center.compute()
            print('Have', center.shape[0], 'centers.')
        **This is deprecated, and using it is horribly discouraged**
        """
        warn('Use of the Stacks context manager is deprecated and may cause pain and sorrow.', DeprecationWarning)
        self.open_stacks(**kwargs)
        yield self.stacks
        self.close_files()

    def add_stack(self, label: str, stack: Union[da.Array, np.ndarray, h5py.Dataset], 
                  overwrite: bool = False, set_diff_stack: bool = False, 
                  persist: bool = True, rechunk: bool = True):
        """Adds a data stack to the data set.
        
        The new data stack can be either a dask array or a numpy array. The only restriction is that its
        first dimension's length (i.e. total number of shots) has to equal the rest of the dataset. The
        stack is *not* stored to disk yet, but it's placed under the control of the dataset object.
        
        If the new data is a numpy array, it will be turned into a dask array with appropriate properties. By
        default (persist=True), it will be eagerly persisted, that is, a copy will be made and the dask graph will
        be simplified.

        Args:
            label (str): Label for the new stack
            stack (Union[da.Array, np.ndarray, h5py.Dataset]): New data stack
            overwrite (bool, optional): Overwrite, if an identically named stack exists already. Defaults to False.
            set_diff_stack (bool, optional): Set the new stack as the 'diffraction data' stack, which will
                recieve some special treatment (e.g. it is never loaded into memory). Defaults to False.
            persist (bool, optional): If the stack is a numpy array, make the dask array persited right away. 
                There is little speaking against it except for some edge cases. Defaults to True.
            rechunk (bool, optional): If the stack is a dask array with a chunk along the first dimension that
                does not match the dataset's overall chunking, rechunk it. This is highly recommended. Defaults to True.
        """

        if stack.shape[0] != self.shots.shape[0]:
            raise ValueError('Stack height must equal that of the shot list.')

        if (label in self._stacks.keys()) and not overwrite:
            raise ValueError(f'Stack with name {label} already exists. Set overwrite = True.')

        if not isinstance(stack, da.Array):
            ch = stack.ndim * [-1]
            ch[0] = self.zchunks          
            if isinstance(stack, np.ndarray) and persist:
                stack = da.from_array(stack, chunks=tuple(ch)).persist(scheduler='threading')
            else:
                stack = da.from_array(stack, chunks=tuple(ch))
        else:
            if (stack.chunks[0] != self.zchunks) and (self.zchunks is not None):
                if rechunk:
                    stack = stack.rechunk({0: self.zchunks})
                else:
                    warn('Stack has a different chunking than the dataset!')

        if set_diff_stack:
            self._diff_stack_label = label

        self._stacks[label] = stack

    def delete_stack(self, label: str, from_files: bool = False):
        """Delete a data stack from the dataset

        Args:
            label (str): label of the stack to delete
            from_files (bool, optional): Also delete stack from the data files. Note that this will
            actually not free up disk space (you need to make a copy of the files for this), and only
            works if the files are open in writable mode. Defaults to False.
        """

        try:
            del self._stacks[label]
        except KeyError:
            warn(f'Stack {label} does not exist, not deleting anything.')

        if label == self._diff_stack_label:
            self._diff_stack_label = ''
            warn(f'Deleting diffraction data stack {label}.', RuntimeWarning)

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
                       include_3d: bool = False, scheduler: Union[str, Client] = 'threading'):
        """Persist the stacks to memory (locally and/or on the cluster workers), that is, they are computed.
        but actually not changed to numpy arrays, just immediately available dask arrays without an actual
        task graph. It is recommended to have as many stacks persisted as possible.
        The diffraction data stack is automatically excluded, as are any 3D arrays (be default).
        
        Note:
            There are important subtleties about which dask scheduler to use here. If you have a 
            dask.distributed cluster running (and you often will), the underlying dask.persist() function if 
            called without parameters will
            compute and persist the data on the *workers* of the cluster, not the local machine. For our typical
            applications (making access to small meta stacks faster and less error-prone), that's the wrong
            choice. Hence, scheduler='threading' by default (you might as well use 'single-threaded'). However,
            there might be cases where persisting on the workers make sense - in that case just set the scheduler
            argument to your client object.

        Args:
            labels (Union[None, str, list], optional): Labels of stacks to persist (None: all except for the one 
                set in diff_stack_label). Defaults to None.
            exclude (Union[None, str, list], optional): Stacks to exclude. Defaults to None.
            include_3d (bool, optional): Include 3D stacks. Defaults to False.
            scheduler (Union[str, Client], optional): What scheduler to use. Defaults to 'threading'.
        """
        
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
            
        exclude.append(self._diff_stack_label)
        
        labels = [l for l in labels if l not in exclude]
        
        print('Persisting stacks to memory:', ', '.join(labels))
        self._stacks.update(dask.persist({sn: stk for sn, stk in self.stacks.items() if stk.ndim < 3}, 
                                         scheduler=scheduler)[0])

    def store_stacks(self, labels: Union[None, str, list] = None, exclude: Union[None, str, list] = None, overwrite: bool = False, 
                    compression: Union[str, int] = 32004, lazy: bool = False, data_pattern: Union[None,str] = None, 
                    progress_bar=True, scheduler: str = 'threading', **kwargs):
        """Stores stacks with given labels to the HDF5 data files. For stacks which are not
        persisted, at this point the actual calculation is done here. 
        
        Note:
            This way of computing
            and storing data is restricted to threading (which does not help much) or single-threaded computation, i.e. 
            it's **not** recommended for heavy lifting, like computing corrected/aggregated/modified diffraction patterns. 
            In this case, better use true parallelism provided by `store_stack_fast`, which uses `dask.distributed` for
            scheduling.

        Args:
            labels (Union[None, str, list], optional): Stacks to be written. If None, write all stacks, *including* 
                the diffraction data stack. Defaults to None.
            exclude (Union[None, str, list], optional): Stacks to exclude. It might be wise to set the diffraction
                data stack here. Defaults to None.
            overwrite (bool, optional): Overwrite existing stacks (HDF5 datasets) in the files. Defaults to False.
            compression (Union[str, int], optional): HDF5 compression filter to use. Common choices are 'gzip', 'none',
                or 32004, which is the lz4 filter often used for diffraction data. Defaults to 32004.
            lazy (bool, optional): Instead of computing and storing the arrays, return a list of dask arrays and HDF5
                data sets, which can be inserted into dask.array.store. Defaults to False.
            data_pattern (Union[None,str], optional): store stacks to this data path (% is replaced by subset) instead 
                of standard data path if not None.
                Note that stacks stored this way will not be retrievable through Dataset objects. Defaults to None.
            progress_bar (bool, optional): show a progress bar during calculation/storing. To prevent a mess,
                disable if you're running store_stacks in multiple processes simultaneously. Defaults to True.
            scheduler (str, optional): dask scheduler to be used. Can be 'threading' or 'single-threaded'. It is not
                possible to use 'multiprocessing' due to conflicting access to HDF5 files. (If you want true parallel
                computation, you have to use `store_stack_fast` instead.) Defaults to 'threading'.
            **kwargs: Will be forwarded to h5py.create_dataset

        Returns:
            None (if lazy=False)
            da.Array, h5py.Dataset: dask arrays and HDF5 dataset to pass to dask.array.store (if lazy=True)
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
            
        labels = [l for l in labels if l not in exclude]
        
        stacks = {k: v for k, v in self._stacks.items() if k in labels}
        stacks.update({'index': da.from_array(self.shots.index.values, chunks=(self.zchunks,))})

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
                                           chunks=(1,) + arr.shape[1:],
                                           compression=compression, **kwargs)
                except (RuntimeError, OSError) as e:
                    if ('name already exists' in str(e)) and (overwrite or label == 'index'):
                        ds = fh.require_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                                chunks=(1,) + arr.shape[1:],
                                                compression=compression, **kwargs)
                    else:
                        print('Cannot write stack', label)
                        raise e
                
                if label == 'index':
                    # print('Writing recommended_zchunks attribute...')
                    fh[path.rsplit('/', 1)[0]].attrs['recommended_zchunks'] = np.array(arr.chunks[0])
                #     fh[path.rsplit('/', 1)[0]].attrs['signal'] = self._diff_stack_label

                arrays.append(arr)
                datasets.append(ds)
                
                if self._diff_stack_label == label:
                    fh[path.rsplit('/',1)[0]].attrs['signal'] = label

        if lazy:
            return arrays, datasets

        else:
            with catch_warnings():
                if progress_bar:
                    with ProgressBar():
                        da.store(arrays, datasets, scheduler=scheduler, return_stored=False)
                else:
                    da.store(arrays, datasets, scheduler=scheduler, return_stored=False)

            for fh in self.file_handles.values():
                fh.flush()
                
    def store_stack_fast(self, label: Optional[str] = None, client: Optional[Client] = None, sync: bool = True,
                         compression: Union[int, str] = 32004) -> pd.DataFrame:
        """Store (and compute) a single stack to HDF5 file(s), using a dask.distributed cluster.
        
        This allows for proper parallel computation (on single or many machines) and is wa(aaa)y faster
        than the standard `store_stacks`, which only works with threads.
        Typically, you'll want to use this method to store a processed diffraction data stack.
            
        Note:
            If the stack to be stored depends on computationally heavy (but memory-fitting) dask
            arrays which you want to retain outside this computation (e.g. to store them using
            store_stacks), make sure they are persisted before calling this function.
            Otherwise, they will be re-calculated from scratch.

        Args:
            label (Optional[str]): Label of the stack to be computed and stored. If None, use the value
                stored in diff_stack_label. Defaults to None
            client (Optional[Client], optional): dask.distributed client connected to a cluster to perform
                the computation on. Defaults to None.
            sync (bool, optional): if True (default), computes and stores immediately, and returns a pandas 
                dataframe containing metadata of everything stored, for validation. If False,
                returns a list of dask.delayed objects which encapsulate the computation/storage. Defaults to True.
            compression (Union[int, str], optional): HDF5 compression filter to use. Common choices are 'gzip', 'none',
                or 32004, which is the lz4 filter often used for diffraction data. Defaults to 32004.

        Returns:
            pd.DataFrame: pandas DataFrame holding ID columns of the computed shots. They can be merged
                with the shot list to cross-check if everything went ok. If sync=False, a list of futures to tuples
                (file, subset, path, idcs) for each dask array chunk is returned instead.
        """

        if label is None:
            label = self._diff_stack_label

        if self._files_open and not self._files_writable:
            raise RuntimeError('Please open files in write mode or close them before storing.')
        
        from distributed import Lock

        stack = self._stacks[label]
            
        print(f'Initializing data sets for diffraction stack {label}...')
            
        # initialize datasets in files
        for (file, subset), grp in self.shots.groupby(['file', 'subset']):
            with h5py.File(file, 'a') as fh:
                path = self.data_pattern.replace('%', subset)
                fh.require_dataset(f'{path}/{label}', 
                                        shape=(len(grp),) + stack.shape[1:], 
                                        dtype=stack.dtype, 
                                        chunks=(1,) + stack.shape[1:], 
                                        compression=compression)
                if label == self._diff_stack_label:
                    fh[path].attrs['signal'] = label
                    
        
        self.close_files()
        
        chunk_label = np.concatenate([np.repeat(ii, cs) for ii, cs in enumerate(stack.chunks[0])])
        stk_del = stack.to_delayed().squeeze()

        locks = {fn: Lock() for fn in self.files}

        dels = []
        print(f'Submitting tasks to dask.distributed scheduler...')
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
            print('Starting computation...')
            chunk_info = client.compute(dels, sync=True)
            return pd.DataFrame(chunk_info, columns=['file', 'subset', 'path', 'shot_in_subset'])
        
        
    def compute_and_save(self, diff_stack_label: Optional[str] = None, list_file: Optional[str] = None, client: Optional[Client] = None, 
                         exclude_stacks: Union[str,List[str]] = None, overwrite: bool = False, persist_diff: bool = True, 
                         persist_all: bool = False, compression: Union[str, int] = 32004,
                         store_features: bool = True):
        """Compound method to fully compute a dataset and write it to disk. 
        
        It is designed for completely writing HDF5 files from scratch, not to append to or modify existing ones,
        in which case you have to use the more fine-grained methods for data storage.
        The foolowing steps are taken:
        
        * Initialize the HDF5 files (using `init_files`)
        * Store the metadata tables (shots, features, peaks, predictions)
        * Compute/store all non-diffraction-data stacks (using `store_stacks`). If this step takes too long, make
          sure that computation-heavy but small stacks are already persisted in memory.
        * Compute/store the diffraction data set (identified by `diff_stack_label`) using `store_stack_fast`.
        * Write a list file which can be used to reload the dataset or to feed into CrystFEL.

        Args:
            diff_stack_label (Optional[str], optional): Label of the diffraction data stack. If None, use
                the one stored in `diff_stack_label`. Defaults to None.
            list_file (Optional[str], optional): Name of the list file to be written. Defaults to None.
            client (Optional[Client], optional): dask.distributed client for computation of the diffraction
                data. Defaults to None.
            exclude_stacks (Union[str,List[str]], optional): Labels of data stacks to exclude. Defaults to None.
            overwrite (bool, optional): Overwrite existing files. Defaults to False.
            persist_diff (bool, optional): Changes the dask array underlying diffraction data stack from the 
                computed one to the one stored in the HDF5 file. This is different from persisting to memory (as is
                done otherwise), as it persists the data *from disk*: if you access it using e.g. .compute(), it will be loaded
                from disk instead of being recomputed. Defaults to True.
            persist_all (bool, optional): Changes dask arrays underlying *all* stacks from the 
                computed one to the one stored in the HDF5 file. Defaults to False.
            compression (Union[str, int], optional): HDF5 compression filter to use. Common choices are 'gzip', 'none',
                or 32004, which is the lz4 filter often used for diffraction data. Defaults to 32004.
            store_features (bool, optional): store/overwrite the feature table into the files. Defaults to True.
        """
        #TODO generalize to storing several diffraction stacks using sync=False.
        
        if diff_stack_label is None:
            diff_stack_label = self.diff_stack_label if self.diff_stack_label else None
         
        if (diff_stack_label is not None) and (diff_stack_label not in self._stacks):
            raise ValueError(f'Stack {diff_stack_label} not found in dataset.')

        if (diff_stack_label is not None) and (client is None):
            raise ValueError(f'If a diffraction data stack is specified, you must supply a dask.distributed client object.')        
            
        for dn in {os.path.dirname(f) for f in self.files}:
            if dn:
                os.makedirs(dn, exist_ok=True)

        exclude_stacks = [exclude_stacks] if isinstance(exclude_stacks, str) else exclude_stacks
        exclude_stacks = [diff_stack_label] if exclude_stacks is None else [diff_stack_label] + exclude_stacks

        print('Initializing data files...')
        self.init_files(overwrite=overwrite)

        print('Storing meta tables...')
        self.store_tables(shots=True, features=store_features)
                
        # store all data stacks except for the actual diffraction data
        self.open_stacks(readonly=False)
        
        meta_stacks = [k for k in self.stacks.keys() if k not in exclude_stacks]
        print(f'Storing meta stacks {", ".join(meta_stacks)}')
        self.store_stacks(labels=meta_stacks, compression=compression, overwrite=overwrite)

        print(f'Storing diffraction data stack {diff_stack_label}... monitor progress at {client.dashboard_link} (or forward port if remote)')
        chunk_info = self.store_stack_fast(diff_stack_label, client, compression=compression)

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
        ss_chunk = self.shots.groupby(['file', 'subset']).size().apply(
            lambda l: ((l // c) * [c]) + ([l % c] if l % c > 0 else []))
        zchunks = np.concatenate([np.array(v) for v in ss_chunk])
        # print(zchunks)
        assert zchunks.sum() == self.shots.shape[0]
        for sn, s in self.stacks.items():
            # print(sn)
            # print(tuple(zchunks))
            self._stacks[sn] = s.rechunk({0: tuple(zchunks)})
            # self.add_stack(sn, s.rechunk({0: tuple(zchunks)}), overwrite=True)

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
            
    def get_indexing_solution(self, stream: Union[str, StreamParser], sol_file: Optional[str] = None, 
                              beam_center: Optional[Union[List, Tuple]] = ('center_x', 'center_y'), 
                              pixel_size: Optional[float] = 0.055, img_size: Union[Tuple, List] = (1556, 516)):
        
        from itertools import product
        
        if isinstance(stream, str):
            stream = StreamParser(stream)
        
        idcol = ['crystal_id', 'region', 'sample', 'run']
        idcol_s = [f'hdf5{self.shots_pattern}/{c}' for c in idcol]
        
        beam_center = list(beam_center) if beam_center is not None else []

        data_col = [''.join(p) for p in 
                        product(('astar_', 'bstar_', 'cstar_'), ('x', 'y', 'z'))] + ['xshift', 'yshift']
                        
        shots = self.shots[['file', 'Event'] + idcol + beam_center]
                        
        sol = shots.merge(stream.shots[idcol_s + data_col], 
            left_on=idcol, right_on=idcol_s, how='left')[['file', 'Event'] + data_col + beam_center].dropna()
        
        if beam_center:
            sol['xshift'] = sol['xshift'] - pixel_size*(sol[beam_center[0]] - img_size[0]//2 + 0.5)
            sol['yshift'] = sol['yshift'] - pixel_size*(sol[beam_center[1]] - img_size[1]//2 + 0.5)
            sol.drop(columns=beam_center, inplace=True)  
        
        if sol_file is not None:
            sol.to_csv(sol_file, header=False, index=False, sep=' ')
        
        return sol
            
    def merge_pattern_info(self, ds_from: Union['Dataset', str], merge_cols: Optional[List[str]] = None, 
                           by: Union[List[str], Tuple[str]] = ('sample', 'region', 'run', 'crystal_id'), 
                           persist: bool = True):
        """Merge shot-table and CXI peak data from another data set into this one, based
        on matching of the shot table columns specified in "by". Default is ('sample', 'region', 'run', 'crystal_id'),
        which matches the shot information based on individual crystals.
        
        The typical application of this function is to take over diffraction pattern information such as pattern center
        and peak positions from an aggregated data set (where each pattern corresponds to exactly one shot) to a
        full data set (where each pattern often corresponds to many shots, such as frames of a diffraction movie).
        
        In this case you'd call the method like: `ds_all.merge_pattern_info(ds_agg)`, where ds_agg is the
        aggregated data set to get the information from.

        Args:
            ds_from (Uniton[Dataset, str]): Diffractem Dataset to take information from, or filename of h5 or list file.
                Esepcially friendly for h5 files written by get_image_info.
            merge_cols (Optional[List[str]], optional): Shot table columns to take over from other data set. If None,
                all columns are taken over which are not present in the shot table currently. Defaults to None.
            by (Union[List[str], Tuple[str]], optional): Shot table columns to match by. 
                Defaults to ('sample', 'region', 'run', 'crystal_id').
            persist (bool, optional): Persist the merged CXI peak data to memory. Defaults to True.
        """
        #TODO Figure out a good way to handle predictions
        
        by = list(by)
        
        if isinstance(ds_from, str):
            ds_from = Dataset.from_files(ds_from, chunking=-1, persist_meta=True)
        
        merge_cols = ds_from.shots.columns.difference(list(self.shots.columns) + 
                                                    ['_Event', '_file', 'file_event_hash']) \
                                                        if merge_cols is None else merge_cols
        
        sh_from = ds_from.shots.copy() # avoid side effects on ds_from
        sh_from['ii_from'] = range(len(sh_from))
        sel_shots = self.shots.merge(sh_from[by + list(merge_cols) + ['ii_from']], on=by, 
                                        how='left', validate='m:1', indicator=True)
        
        if not all(sel_shots._merge == 'both'):
            print(sel_shots.query('_merge != "both"')[['file', 'Event', 'region', 'crystal_id']])
            raise ValueError('Not all features present in the dataset are present in ds_from.')

        self.shots = sel_shots.drop('_merge', axis=1)

        peakdata = ds_from.peak_data

        if not all(self.shots.ii_from.diff().fillna(1) == 1):
            peakdata = {k: v[self.shots.ii_from.values,...] for k, v in peakdata.items()}

        self.peak_data = peakdata
        # for k, v in peakdata.items():
        #     self.add_stack(k, v, overwrite=True)
        
        if persist: 
            self.persist_stacks(list(peakdata))

    def merge_acquisition_data(self, fields: dict):
        # mange instrument (acquisition) data like exposure time etc. into shot list
        raise NotImplementedError('merge_acquisition_data not yet implemented')

    def write_list(self, listfile: str, append: bool = False):
        """
        Writes the files in the dataset into a list file, containing each file on a line.
        
        Args:
            listfile (str): list file name
        """
        #TODO allow to export CrystFEL-style single-pattern lists
        with open(listfile, 'a' if append else 'w') as fh:
            fh.write('\n'.join(self.files) + '\n')

    def write_virtual_file(self, filename: str = 'virtual', diff_stack_label: str = 'zero_image',
                              virtual_size: int = 1024):
        """
        Generate a virtual HDF5 file containing the meta data of the dataset, but not the actual
        diffraction. Instead of the diffraction stack, a dummy stack containing only ones is written
        into the file, which due to its compression becomes very small.
        
        The peak positions in the virtual file are changed, such that they refer to a "virtual" geometry,
        corresponding to a square detector with a size given by `virtual_size`. On this detector, the pattern
        is centered.
        
        This file can then be used as input to CrystFEL's *indexamajig*, with a simple centered geometry.

        Args:
            filename (str): [description]
            diff_stack_label (str): [description]
            virtual_size (int, optional): [description]. Defaults to 1024.

        """
        self._shot_id_cols
        ds_ctr = self.get_selection('True', file_suffix='_virtual.h5', new_folder='')
        # ds_ctr.shots['file_event_hash'] = tools.dataframe_hash(self.shots[['file', 'Event']])
        # ds_ctr.shots['feature_hash'] = tools.dataframe_hash(self.shots[['sample', 'region', 'run', 'crystal_id']])
        ds_ctr.shots[['_file', '_Event']] = self.shots[['file', 'Event']]
        ds_ctr.shots['file'] = f'{filename}.h5'
        ds_ctr.shots['subset'] = 'entry'
        ds_ctr.shots['shot_in_subset'] = range(len(ds_ctr.shots))
        ds_ctr.shots['Event'] = ds_ctr.shots.subset + '//' + ds_ctr.shots.shot_in_subset.astype(str)

        fake_img = da.ones(dtype=np.int8, shape=(ds_ctr.shots.shape[0], virtual_size, virtual_size), 
                            chunks=(1, -1, -1))

        ds_ctr.add_stack(diff_stack_label, fake_img, overwrite=True, set_diff_stack=True)
        ds_ctr.add_stack('peakXPosRaw', (ds_ctr.peakXPosRaw - self.shots.center_x.values.reshape(-1,1) 
                                        + virtual_size/2 - 0.5) 
                        * (ds_ctr.peakXPosRaw != 0), overwrite=True)
        ds_ctr.add_stack('peakYPosRaw', (ds_ctr.peakYPosRaw - self.shots.center_y.values.reshape(-1,1) 
                                        + virtual_size/2 - 0.5) 
                        * (ds_ctr.peakYPosRaw != 0), overwrite=True)

        print('Writing fake all-ones data (yes, it takes that long).')
        with h5py.File(ds_ctr.files[0], 'w') as fh:
            fh.require_group('/entry/data')
        ds_ctr.open_stacks(readonly=False)
        ds_ctr.store_stacks([diff_stack_label, 'nPeaks', 'peakXPosRaw', 'peakYPosRaw', 
                            'peakTotalIntensity'],
                            compression='gzip', overwrite=True)
        ds_ctr.close_stacks()
        ds_ctr.store_tables(shots=True, features=False, peaks=False, predict=False)
        ds_ctr.write_list(f'{filename}.lst')
        print(f'Virtual file {filename}.h5 and list file {filename}.lst successfully exported.')
        
    def view(self, **kwargs):
        """Calls `tools.viewing_widget` on the dataset. Keyword arguments are passed through.
        """
        tools.viewing_widget(self, **kwargs)