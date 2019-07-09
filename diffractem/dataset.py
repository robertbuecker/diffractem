import pandas as pd
import dask.array as da
import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
from diffractem.io import expand_files, get_meta_lists, get_data_stacks
from diffractem.nexus import get_table, get_meta_fields
from . import io, nexus
from .stream_parser import StreamParser
from .map_image import MapImage
import h5py
from typing import Union, List, Dict
import copy
from collections import defaultdict
from warnings import warn, catch_warnings, simplefilter
from tables import NaturalNameWarning


class Dataset:

    def __init__(self):

        self._shots_changed = False
        self._peaks_changed = False
        self._predict_changed = False
        self._features_changed = False

        # HDF5 file addresses
        self.data_pattern = '/%/data'
        self.result_pattern = '/%/results'
        self.map_pattern = '/%/map'
        self.instrument_pattern = '/%/instrument'
        self.parallel_io = True

        # internal stuff
        self._h5handles = {}
        self._zchunks = 1
        self._stacks = {}
        self._shot_id_cols = ['file', 'Event']
        self._feature_id_cols = ['crystal_id', 'region', 'sample']
        self._stacks_open = False

        # tables
        self._shots = pd.DataFrame(columns=self._shot_id_cols + self._feature_id_cols + ['selected'])
        self._peaks = pd.DataFrame(columns=self._shot_id_cols)
        self._predict = pd.DataFrame(columns=self._shot_id_cols)
        self._features = pd.DataFrame(columns=self._feature_id_cols)

    def __str__(self):
        return (f'diffractem Dataset object spanning {len(self._shots.file.unique())} NeXus/HDF5 files\n-----\n'
                f'{self._shots.shape[0]} shots ({self._shots.selected.sum()} selected)\n'
                f'{self._peaks.shape[0]} peaks, {self._predict.shape[0]} predictions, '
                f'{self._features.shape[0]} features\n'
                f'{len(self._stacks)} data stacks: {list(self._stacks.keys())}\n'
                f'Data stacks open: {self._stacks_open}\n')

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
        return self._predict

    @property
    def features(self):
        return self._features

    @property
    def peaks(self):
        return self.get_peaks(False)

    @peaks.setter
    def peaks(self, value):
        self._peaks = value
        self._peaks_changed = True

    def get_peaks(self, selected=False):
        if selected:
            return self._sel(self._peaks)
        else:
            return self._peaks

    @predict.setter
    def predict(self, value):
        self._predict = value
        self._predict_changed = True

    @features.setter
    def features(self, value):
        self._features = value
        self._features_changed = True

    @classmethod
    def from_list(cls, listfile: Union[list, str], init_stacks=True, load_tables=True, **kwargs):
        """
        Creates a data set from a .lst file, which contains a simple list of H5 files (on separate lines).
        Alternatively, accepts a single filename, or even a python list of files. If the .lst file has CrystFEL-style
        event indicators in it, it will be loaded, and the events present in the list will be selected, the others not.
        :param listfile: list file name, H5 file name, or list
        :param opts: attributes to be set right away
        :return: dataset object
        """

        file_list = io.expand_files(listfile, scan_shots=True)
        print(file_list)
        self = cls()

        for k, v in kwargs.items():
            self.__dict__[k] = v

        self.load_tables(shots=True, files=list(file_list.file.unique()))

        # now set selected property...
        if 'Event' in file_list.columns:
            self._shots['selected'] = self._shots[self._shot_id_cols].isin(file_list[self._shot_id_cols]).all(axis=1)

        # and initialize stacks and tables
        if init_stacks:
            self.init_stacks()
        if load_tables:
            self.load_tables(features=True, peaks=True, predict=True)

        return self

    def load_tables(self, shots=False, features=False, peaks=False, predict=False, files=None):
        """
        Load pandas metadata tables from the HDF5 files. Set the argument for the table you want to load to True.
        :param shots:
        :param features:
        :param peaks:
        :param predict:
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
                self._shots = nexus.get_table(files, self.data_pattern + '/shots',
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
                print('No shots found at ' + self.data_pattern + '/shots')

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
                print('No peaks found at ' + self.result_pattern + '/peaks')

        if predict:
            try:
                self._predict = nexus.get_table(files, self.result_pattern + '/predict', parallel=self.parallel_io)
                self._predict_changed = False
            except KeyError:
                print('No predictions found at ' + self.result_pattern + '/predict')

    def store_tables(self, shots: Union[None, bool] = None, features: Union[None, bool] = None,
                     peaks: Union[None, bool] = None, predict: Union[None, bool] = None):
        """
        Stores the metadata tables (shots, features, peaks, predictions) into HDF5 files. For each of the tables,
        it can be automatically determined if they have changed and should be stored...
        HOWEVER, this only works if no inplace changes have been made. So don't rely on it too much.
        :param shots: True -> store shot list, False -> don't store shot list, None -> only store if changed
        :param features: similar
        :param peaks: similar
        :param predict: similar
        :param parallel_io: write to HDF5 files in parallel. Usually works well and is way faster.
        :return:
        """
        fs = []

        if self._stacks_open:
            warn('Data stacks are open, and will be transiently closed. You will need to re-create derived stacks.',
                 RuntimeWarning)
            stacks_were_open = True
            self.close_stacks()
        else:
            stacks_were_open = False

        simplefilter('ignore', NaturalNameWarning)
        if (shots is None and self._shots_changed) or shots:
            fs.extend(nexus.store_table(self.shots, self.data_pattern + '/shots', parallel=self.parallel_io))
            self._shots_changed = False

        if (features is None and self._features_changed) or features:
            fs.extend(nexus.store_table(self.features, self.map_pattern + '/features', parallel=self.parallel_io))
            self._features_changed = False

        if (peaks is None and self._peaks_changed) or peaks:
            fs.extend(nexus.store_table(self.peaks, self.result_pattern + '/peaks', parallel=self.parallel_io))
            self._peaks_changed = False

        if (predict is None and self._predict_changed) or predict:
            fs.extend(nexus.store_table(self.predict, self.result_pattern + '/predict', parallel=self.parallel_io))
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

        cols = list(self.shots.columns.difference(stream.shots.columns)) + self._shot_id_cols
        self.shots = self.shots[cols].merge(stream.shots,
                                            on=self._shot_id_cols, how='left', validate='1:1')
        self.peaks = stream.peaks.merge(self.shots[self._shot_id_cols + ['subset', 'shot_in_subset']],
                                        on=self._shot_id_cols, how='inner')
        self.predict = stream.indexed.merge(self.shots[self._shot_id_cols + ['subset', 'shot_in_subset']],
                                            on=self._shot_id_cols, how='inner')

    def get_map(self, file, subset='entry') -> MapImage:
        # get a MapImage from stored data, with tables filled in from dataset
        map = MapImage()
        return map

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

    @property
    def _tables(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the raw tables (_shots, _peaks, _predict, _features) as dict. Mostly for internal or very careful use.
        :return: dict with raw tables
        """
        lbls = ['_shots', '_peaks', '_predict', '_features']
        return {k: v for k, v in self.__dict__.items() if k in lbls}

    def change_filenames(self, file_suffix: str = '.h5', file_prefix: str = '',
                         new_folder: Union[str, None] = None,
                         fn_map: Union[pd.DataFrame, None] = None,
                         keep_raw=True):
        """
        Change file names in all lists using some handy modifications. The old file names are copied to a "file_raw"
        column, if not already present (can be overriden with keep_raw).
        :param file_suffix: add suffix to file, INCLUDING file extension, e.g. '_modified.h5'
        :param file_prefix: add suffix to actual filenames (not folder/full path!), e.g. 'aggregated_
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
            new_fn = file_prefix + folder_file[1].str.rsplit('.', 1, expand=True)[0] + file_suffix
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

    def init_files(self, overwrite=False, parallel=False):
        """
        Make new files corresponding to the shot list, by copying over instrument metadata and maps (but not
        results, shot list, data arrays,...
        :param overwrite: overwrite new files if not yet existing
        :param parallel: copy file data over in parallel. Can be way faster.
        :return:
        """
        fn_map = self.shots[['file', 'file_raw']].drop_duplicates()
        for _, filepair in fn_map.iterrows():
            if parallel:
                raise NotImplementedError('parallel: later')
            else:
                print(filepair['file_raw'], ' -> ', filepair['file'])
                nexus.copy_h5(filepair['file_raw'], filepair['file'], mode='w' if overwrite else 'w-',
                              exclude=('%/detector/data', self.data_pattern + '/%', self.result_pattern + '/%'),
                              print_skipped=False)

    def get_selection(self, query: Union[str, None] = None,
                      file_suffix: str = '_sel.h5', file_prefix: str = '',
                      new_folder: Union[str, None] = None) -> 'Dataset':
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
            newset.reset_id()
            newset._h5handles = {}

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

    def aggregate(self, by: Union[list, tuple] = ('run', 'region', 'crystal_id', 'sample'),
                  how: Union[dict, str] = 'mean',
                  file_suffix: str = '_agg.h5', file_prefix: str = '', new_folder: Union[str, None] = None,
                  rechunk: Union[None, int] = None, allow_mix: bool = False) -> 'Dataset':
        """
        Aggregate sub-sets of stacks using different aggregation functions. Typical application: sum sub-stacks of
        dose fractionation movies, or shots with different tilt angles (quasi-precession)
        :param by:
        :param how:
        :param file_suffix:
        :param file_prefix:
        :param new_folder:
        :param rechunk:
        :param allow_mix:
        :return:
        """

        by = list(by)
        # aggregate shots
        newset = copy.copy(self)
        agg_stacks = {k: [] for k in self.stacks.keys()}
        agg_shots = []

        if not self._stacks_open:
            raise RuntimeError('Stacks are not open.')

        for _, grp in self.shots.groupby(by):
            # loop over aggregation group

            # investigate the group a bit...
            if all(np.diff(grp.index) == 1):
                idcs = slice(grp.index[0], grp.index[-1] + 1) # aggregation is continuous region in shot list
                continuous = True
            else:
                idcs = grp.index.values  # aggregation is a non-continuous region in shot list
                continuous = False

            # DATA STACKS ----

            for sn, s in self.stacks.items():
                method = how if isinstance(how, str) else how.get(sn, 'mean')

                if method == 'sum':
                    newstack = s[idcs, ...].sum(axis=0, keepdims=True)
                elif method in ['mean', 'average', 'avg']:
                    newstack = s[idcs, ...].mean(axis=0, keepdims=True)
                elif method == 'cumsum':
                    newstack = s[idcs, ...].cumsum(axis=0)
                else:
                    raise ValueError(f'Unknown aggregation method {how}')

                agg_stacks[sn].append(newstack)

            newset._stacks = {sn: da.concatenate(s) for sn, s in agg_stacks.items()}

            # SHOT LIST ----

            newshots = grp.copy()

            # find columns with non-identical entries in the aggregation group
            nunique = newshots.apply(pd.Series.nunique)
            cols_nonid = nunique[nunique > 1].index

            if ('file' in cols_nonid) or ('subset' in cols_nonid):
                # subset and file are different within aggregation. #uh-oh #crazydata #youpunkorwhat
                # ...trying to come up with a sensible file and subset name, by finding common sequences

                if not allow_mix:
                    raise ValueError('Aggregation of data from different files requested. Set allow_mix to true to'
                                     'allow that.')
                from functools import reduce
                from difflib import SequenceMatcher

                common_file = reduce(lambda s1, s2: ''.join([s1[m.a:m.a + m.size] for m in
                                                             SequenceMatcher(None, s1, s2).get_matching_blocks()]),
                                     newshots.files)
                common_subset = reduce(lambda s1, s2: ''.join([s1[m.a:m.a + m.size] for m in
                                                               SequenceMatcher(None, s1, s2).get_matching_blocks()]),
                                       newshots.subset)
                newshots.file = common_file
                newshots.subset = common_subset

            # ID string for some explanation
            newshots['aggregation'] = newshots.iloc[0]['file'] + ':' + newshots.iloc[0]['Event'] + '->' + \
                                      newshots.iloc[-1]['file'] + ':' + newshots.iloc[-1]['Event']

            if newshots.shape[0] > 1:
                newshots = pd.concat([newshots.iloc[0:1, :]] * newstack.shape[0], axis=0, ignore_index=True)
            agg_shots.append(newshots)

        print(cols_nonid)

        newset._shots = pd.concat(agg_shots, axis=0, ignore_index=True)
        newset._features = self._features.merge(newset._shots[self._feature_id_cols],
                                                on=self._feature_id_cols, how='inner', validate='1:m'). \
            drop_duplicates(self._feature_id_cols).reset_index(drop=True)

        newset._h5handles = {}
        newset.change_filenames(file_suffix, file_prefix, new_folder)
        newset.reset_id(keep_raw=False)

        # drop remaining columns that are inconsistent
        for col in cols_nonid:
            if col not in self._shot_id_cols + ['shot_in_subset']:
                newset._shots.drop(col, axis=1)

        return newset

    def init_stacks(self):
        """
        Opens stacks briefly, to check their sizes etc., and closes them again right away. Helpful to just get their
        names and sizes...
        :return:
        """
        self.open_stacks(init=True)
        self.close_stacks()

    def close_stacks(self):
        """
        Close the stacks by closing the HDF5 data file handles. Technically, this does not delete the dask arrays,
        but you are blocked from accessing them. Any changes made to the arrays _and_ anything derived from them
        are discarded, so handle with great care.
        :return:
        """
        for f in self._h5handles.values():
            f.close()
        self._stacks_open = False

    def open_stacks(self, labels: Union[None, list] = None, checklen=True, init=False):
        """
        Opens data stacks from HDF5 (NeXus) files (found by the "data_pattern" attribute), and assigns dask array
        objects to them. After opening, the arrays or parts of them can be accessed through the stacks attribute,
        or directly using a dataset.stack syntax, and loaded using the .compute() method of the arrays.
        Note that, while the stacks are open, no other python kernel will be able to access the data files, and other
        codes may not see the modifications made. You will have to call dataset.close_stacks() before.
        :param labels: list of stacks to open. Default: None -> opens all stacks
        :param checklen: check if stack heights (first dimension) is equal to shot list length
        :param init: do not load stacks, just make empty dask arrays. Usually the init_stacks method is more useful.
        :return:
        """
        sets = self._shots[['file', 'subset', 'shot_in_subset']].drop_duplicates()
        stacks = defaultdict(list)

        for (fn, subset), subgrp in sets.groupby(['file', 'subset']):
            self._h5handles[fn] = fh = h5py.File(fn)
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
                        if init:
                            newstack = da.empty(shape=ds.shape, dtype=ds.dtype, chunks=-1)
                        else:
                            # print('adding stack: '+ds.name)
                            newstack = da.from_array(ds, chunks=ds.chunks)
                        stacks[dsname].append(newstack)

        self._stacks.update({sn: da.concatenate(s, axis=0) for sn, s in stacks.items()})
        self._stacks_open = True

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
            ch[0] = self._zchunks
            stack = da.from_array(stack, chunks=tuple(ch))

        self._stacks[label] = stack

    def store_stacks(self, labels: Union[None, list] = None, overwrite=False, compression=32004, lazy=False, **kwargs):
        """
        Stores stacks with given labels to the HDF5 data files. If None (default), stores all stacks. New stacks are
        typically not yet computed, so at this point the actual data crunching is done.
        :param labels: stacks to be written
        :param overwrite: overwrite stacks already existing in the files?
        :param compression: compression algorithm to be used. 32004 corresponds to bz4, which we mostly use.
        :param lazy: if True, instead of writing the shots, returns two lists containing the arrays and dataset objects
                        which can be used to later pass them to dask.array.store. Default False (store right away)
        :param **kwargs: will be forwarded to h5py.create_dataset
        :return:
        """

        if not self._stacks_open:
            raise RuntimeError('Please open stacks before storing.')

        if labels is None:
            labels = self._stacks.keys()

        stacks = {k: v for k, v in self._stacks.items() if k in labels}
        stacks.update({'index': da.from_array(self.shots.index.values, chunks=(1,))})

        datasets = []
        arrays = []

        shots = self._shots.reset_index()  # just to be safe

        for (fn, ssn), sss in shots.groupby(['file', 'subset']):
            fh = self._h5handles[fn]

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
                path = self.data_pattern.replace('%', ssn) + '/' + label

                try:
                    ds = fh.create_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                           chunks=tuple([c[0] for c in arr.chunks]),
                                           compression=compression, **kwargs)
                except RuntimeError as e:
                    if overwrite or label == 'index':
                        ds = fh.require_dataset(path, shape=arr.shape, dtype=arr.dtype,
                                                chunks=tuple([c[0] for c in arr.chunks]),
                                                compression=compression, **kwargs)
                    else:
                        raise e

                arrays.append(arr)
                datasets.append(ds)

        if lazy:
            return arrays, datasets

        else:
            with catch_warnings():
                with ProgressBar():
                    da.store(arrays, datasets)

            for fh in self._h5handles.values():
                fh.flush()

    def stack_to_shots(self, labels: Union[str, list], selected=False):
        # mangle stack data into the shot list
        pass

    def merge_acquisition_data(self, fields: dict):
        # mange instrument (acquisition) data like exposure time etc. into shot list
        pass


def reduce_stack(filename, first_frame=1, last_frame=-1, aggregate='sum', suffix=None, exclude=(),
                 threads=True, instrument_data=True, label='raw_counts', **kwargs):
    exp_time_field = '/entry/instrument/detector/collection/shutter_time'
    tilt_field = '/entry/instrument/Stage/A'
    if last_frame == -1:
        last_frame = 16383
    exclude = [f'frame < {first_frame}', f'frame > {last_frame}'] + list(exclude)

    from multiprocessing.pool import ThreadPool

    def make_stacks(raw_name):
        sr0 = get_nxs_list(raw_name, 'shots')

        if instrument_data:
            instrument_data = pd.DataFrame(get_meta_fields(raw_name, [exp_time_field, tilt_field])).astype(float)
            instrument_data.columns = ['exp_time', 'tilt_angle']
            instrument_data['tilt_angle'] = (instrument_data['tilt_angle'] * 180 / np.pi).round(1)
            sr0 = sr0.merge(instrument_data, right_index=True, left_on='file')

        sr0['selected'] = True
        for ex in exclude:
            sr0.loc[sr0.eval(ex), 'selected'] = False

        return modify_stack(raw_name, sr0, aggregate=aggregate, labels=label, **kwargs)

    if threads:
        with ThreadPool() as pool:
            stkdat = pool.map(make_stacks, expand_files(filename))
    else:
        stkdat = make_stacks(expand_files(filename))

    shots = pd.concat([s[1] for s in stkdat if s[1].shape[0]], ignore_index=True)
    stack_raw = da.concatenate([s[0][label] for s in stkdat if s[0][label].shape[0]], axis=0)
    shots['file_raw'] = shots['file']

    return None


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

    if shot_list_final.shape[0] == 0:
        stacks_final = {sn: da.from_array(np.zeros((0,) + s.shape[1:]), chunks=-1) for sn, s in stacks.items()}
        return stacks_final, shot_list_final

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
                elif aggregate in 'cumsum':
                    newstack = s[grp.index.values, ...].cumsum(axis=0)
                else:
                    raise ValueError('aggregate must be sum or mean')
                if newchunk is not None:
                    # TODO this does not seem to do anything? Why?!
                    newstack = da.rechunk(newstack, (newchunk, -1, -1))

                agg_stacks[sn].append(newstack)
                agg_shots.append(grp.iloc[:newstack.shape[0], :])  # at this point, they all should be the same

        shot_list_final = pd.concat(agg_shots, ignore_index=True)
        print(f'Aggregated: {shot_list_final.shape[0]} shots')

        stacks_final = {sn: da.concatenate(s) for sn, s in agg_stacks.items()}

    else:
        stacks_final = {sn: s[shot_list_final.index.values, ...] for sn, s in stacks.items()}
        shot_list_final.reset_index(drop=True, inplace=True)

    shot_list_final['shot_in_subset'] = shot_list_final.groupby(['file', 'subset']).cumcount()

    return stacks_final, shot_list_final