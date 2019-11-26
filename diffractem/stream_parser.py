import pandas as pd
from io import StringIO
import numpy as np
import subprocess
import re

BEGIN_GEOM = '----- Begin geometry file -----'
END_GEOM = '----- End geometry file -----'
BEGIN_CELL = '----- Begin unit cell -----'
END_CELL = '----- End unit cell -----'
BEGIN_CHUNK = '----- Begin chunk -----'
END_CHUNK = '----- End chunk -----'
BEGIN_CRYSTAL = '--- Begin crystal'
END_CRYSTAL = '--- End crystal'
BEGIN_PEAKS = 'Peaks from peak search'
END_PEAKS = 'End of peak list'
BEGIN_REFLECTIONS = 'Reflections measured after indexing'
END_REFLECTIONS = 'End of reflections'
HEAD = 'CrystFEL stream format {}.{}'.format(2, 3)
GENERATOR = 'Generated by diffractem StreamParser'
PEAK_COLUMNS = ['fs/px', 'ss/px', '(1/d)/nm^-1', 'Intensity', 'Panel', 'stream_line']
REFLECTION_COLUMNS = ['h', 'k', 'l', 'I', 'Sigma(I)', 'Peak', 'Background', 'fs/px', 'ss/px', 'Panel', 'stream_line']
ID_FIELDS = ['file', 'Event', 'serial']

def chop_stream(streamname: str, id_list: list, id_field='hdf5/%/shots/frame', id_appendix='frame'):
    """Chops a stream file into sub-streams containing only shots with a specific value of
    a defined field, which must be in the chunk header. Useful e.g. for chopping into aggregation
    frames.
    
    Arguments:
        streamname {str} -- [Stream file name]
        id_list {str} -- [List of values of the ID variable which you want to have in the final files.]
    
    Keyword Arguments:
        id_field {str} -- [Field in chunk data to select by] (default: {'hdf5/%/shots/frame'})
        id_appendix {str} -- [file appendix specifying the ID] (default: {'frame'})
    
    Raises:
        RuntimeError: [weirdness in the stream found]
    """

    outfiles = {}
    for fnum in id_list:
        outfiles[fnum] = open(streamname.rsplit('.',1)[0] + f'-{id_appendix}{fnum}.stream', 'w')
    
    chunk_init = False
    chunk_string = ''
    frame = -1
    
    with open(streamname, 'r') as fh_in:
        for ln, l in enumerate(fh_in):
        
            if not chunk_init and l.startswith(BEGIN_CHUNK):
                chunk_init = True
                chunk_string += l
                frame = -1
                
            elif chunk_init and l.startswith(id_field):
                found_frame = int(l.rsplit('=',1)[-1].strip())
                chunk_string += l
                frame = found_frame if found_frame in id_list else -1
        
            elif chunk_init and l.startswith(END_CHUNK):
                chunk_init = False
                chunk_string += l
                #print(frame)
                if frame != -1:
                    #print(chunk_string)
                    outfiles[frame].write(chunk_string)
                chunk_string = ''
                
            elif chunk_init:
                chunk_string += l
        
            elif not chunk_init:
                # no chunk initialized, write to all files
                for _, fh in outfiles.items():
                    fh.write(l)
                    
            else:
                raise RuntimeError('This should not happen?! Please debug me.')

def parse_str_val(input: str):
    try:
        return int(input.strip())
    except ValueError:
        try:
            return float(input.strip())
        except:
            return input.strip()

class StreamParser:

    def __init__(self, filename, parse_now=True, serial_offset=-1, new_folder=None):

        self.merge_shot = False
        self.command = ''
        self._cell_string = []
        self._geometry_string = []
        self._peaks = pd.DataFrame()
        self._indexed = pd.DataFrame()
        self._shots = pd.DataFrame()
        self._crystals = pd.DataFrame()
        self._parsed_lines = 0
        self._total_lines = 0
        self.filename = filename
        self.serial_offset = serial_offset

        if parse_now:
            self.parse(new_folder)

    @property
    def geometry(self):
        """

        :return: geometry section as dictionary
        """

        g = {}
        for l in self._geometry_string:
            if '=' not in l:
                continue
            k, v = l.split('=', 1)
            g[k.strip()] = parse_str_val(v)

        return g

    @property
    def cell(self):
        """

        :return: cell section as dictionary
        """

        c = {}
        for l in self._cell_string:
            if '=' not in l:
                continue
            k, v = l.split('=', 1)
            try:
                c[k.strip()] = float(v)
            except ValueError:
                c[k.strip()] = v.strip()

        return c

    @property
    def options(self):
        """

        :return: crystfel call options (ONLY -- ones) as dict
        """
        o = {}
        for opt in re.findall('--\S+', self.command):
            if '=' in opt:
                k, v = opt[2:].split('=', 1)
                try:
                    o[k.strip()] = int(v)
                except ValueError:
                    try:
                        o[k.strip()] = float(v)
                    except ValueError:
                        o[k.strip()] = v.strip()
            else:
                o[opt[2:].strip()] = None
        return o

    @property
    def indexed(self):
        return self._indexed

    @property
    def peaks(self):
        return self._peaks

    @property
    def shots(self):
        return self._shots.merge(self._crystals, on=ID_FIELDS, how='left')

    @property
    def input_file(self):
        return self.command.split('-i ')[1].split(' -')[0].strip()

    @property
    def files(self):
        return list(self.shots.file.unique())

    def parse(self, new_folder):

        linedat_peak = StringIO()
        linedat_index = StringIO()
        shotlist = []
        crystallist = []
        init_peak = False
        init_index = False
        init_geom = False
        init_cell = False
        init_crystal_info = False
        init_chunk = False
        shotdat = {'Event': None, 'shot_in_subset': None, 'subset': None,
                   'file': None, 'serial': None}
        crystal_info = {}
        idstr = None
        self._parsed_lines = 0
        self._total_lines = 0
        skip = False

        # lines are queried for their meaning. Lines belonging to tables are appended to StringIO virtual files,
        # which are then read into pandas data frames at the very end. The order of Queries is chosen to optimize
        # performance, that is, the table lines (most frequent) come first.
        with open(self.filename) as fh:

            for ln, l in enumerate(fh):

                self._parsed_lines += 1
                self._total_lines += 1
                if skip:
                    skip = False
                    continue

                # EVENT CHUNKS

                # Actual parsing (indexed peaks)
                if init_index and END_REFLECTIONS in l:
                    init_index = False
                elif init_index:
                    linedat_index.write(
                        ' '.join([l.strip(), str(ln), idstr, '\n']))

                # Actual parsing (found peaks)
                elif init_peak and END_PEAKS in l:
                    init_peak = False
                elif init_peak:
                    linedat_peak.write(
                        ' '.join([l.strip(), str(ln), idstr, '\n']))

                # Required info at chunk head
                elif BEGIN_CHUNK in l:
                    shotdat = {'Event': '_', 'shot_in_subset': -1, 'subset': '_',
                               'file': '', 'serial': -1, 'first_line': ln, 'last_line': -1}
                    init_chunk = True
                elif END_CHUNK in l:
                    shotdat['last_line'] = ln
                    shotlist.append(shotdat)
                    shotdat = {'Event': None, 'shot_in_subset': None, 'subset': None,
                               'file': None, 'serial': None, 'first_line': None, 'last_line': None}
                    init_chunk = False
                elif 'Event:' in l:
                    shotdat['Event'] = l.split(': ')[-1].strip()
                    dummy_shot = shotdat['Event'].split('//')[-1]
                    if dummy_shot == '_':
                        shotdat['shot_in_subset'] = 0
                    else:
                        shotdat['shot_in_subset'] = int(shotdat['Event'].split('//')[-1])
                    shotdat['subset'] = shotdat['Event'].split('//')[0].strip()
                elif 'Image filename:' in l:
                    shotdat['file'] = l.split(':')[-1].strip()
                    if new_folder is not None:
                        shotdat['file'] = new_folder + '/' + shotdat['file'].rsplit('/', 1)[-1]
                elif 'Image serial number:' in l:
                    shotdat['serial'] = int(l.split(': ')[1]) + self.serial_offset
                elif (' = ' in l) and (not init_crystal_info) and init_chunk:    # optional shot info
                    k, v = l.split(' = ', 1)                           
                    shotdat[k.strip()] = parse_str_val(v)

                # Table parsing activation for found peaks
                elif (None not in shotdat.values()) and (BEGIN_PEAKS in l):
                    skip = True # skip the column header line
                    init_peak = True
                    idstr = ' '.join([shotdat['file'], shotdat['Event'], str(shotdat['serial'])])

                # Table parsing activation for indexing
                elif (None not in shotdat.values()) and (BEGIN_REFLECTIONS in l):
                    skip = True
                    init_index = True
                    idstr = ' '.join([shotdat['file'], shotdat['Event'], str(shotdat['serial'])])

                # Additional information from indexing
                elif BEGIN_CRYSTAL in l:
                    crystal_info = {k: shotdat[k] for k in ID_FIELDS}
                    init_crystal_info = True
                elif END_CRYSTAL in l:
                    crystallist.append(crystal_info)
                    crystal_info = {}
                    init_crystal_info = False
                elif 'Cell parameters' in l:
                    for k, v in zip(['a', 'b', 'c', 'dummy', 'al', 'be', 'ga'], l.split(' ')[2:9]):
                        if k == 'dummy':
                            continue
                        crystal_info[k] = float(v)
                elif 'astar' in l:
                    crystal_info.update({k: float(v) for k, v in zip(['astar_x', 'astar_y', 'astar_z'], l.split(' ')[2:5])})
                elif 'bstar' in l:
                    crystal_info.update({k: float(v) for k, v in zip(['bstar_x', 'bstar_y', 'bstar_z'], l.split(' ')[2:5])})
                elif 'cstar' in l:
                    crystal_info.update({k: float(v) for k, v in zip(['cstar_x', 'cstar_y', 'cstar_z'], l.split(' ')[2:5])})
                elif 'diffraction_resolution_limit' in l:
                    crystal_info['diff_limit'] = float(l.rsplit(' nm', 1)[0].rsplit('= ', 1)[-1])
                elif 'predict_refine/det_shift' in l:
                    crystal_info['xshift'] = float(l.split(' ')[3])
                    crystal_info['yshift'] = float(l.split(' ')[6])
                    continue
                elif (' = ' in l) and init_crystal_info and init_chunk:    # optional shot info
                    k, v = l.split(' = ', 1)  
                    crystal_info[k.strip()] = parse_str_val(v)

                # CALL STRING

                elif 'indexamajig' in l:
                    self.command = l

                # GEOMETRY FILE

                elif init_geom and (END_GEOM in l):
                    init_geom = False
                elif init_geom:
                    self._geometry_string.append(l.strip())
                elif BEGIN_GEOM in l:
                    init_geom = True

                # CELL FILE

                elif init_cell and (END_CELL in l):
                    init_cell = False
                elif init_cell:
                    self._cell_string.append(l.strip())
                elif BEGIN_CELL in l:
                    init_cell = True

                else:
                    self._parsed_lines -= 1


        # Now convert to pandas data frames

        linedat_index.seek(0)
        linedat_peak.seek(0)
        self._peaks = pd.read_csv(linedat_peak, delim_whitespace=True, header=None,
                              names=PEAK_COLUMNS + ['file', 'Event', 'serial']
                              ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(
            drop=True).drop('index', axis=1)

        self._indexed = pd.read_csv(linedat_index, delim_whitespace=True, header=None,
                               names=REFLECTION_COLUMNS + ['file', 'Event', 'serial']
                               ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(
            drop=True).drop('index', axis=1)

        self._shots = pd.DataFrame(shotlist).sort_values('serial').reset_index(drop=True)
        if crystallist:
            self._crystals = pd.DataFrame(crystallist).sort_values('serial').reset_index(drop=True)
        else:
            self._crystals = pd.DataFrame(columns=ID_FIELDS)

    def write(self, filename, include_peaks=True, include_indexed=True, include_geom=True, include_cell=True):

        with open(filename, 'w') as fh:
            fh.write(HEAD+'\n'+GENERATOR+'\n'+self.command+'\n')
            if include_geom:
                fh.write(BEGIN_GEOM+'\n'+'\n'.join(self._geometry_string)+'\n'+END_GEOM + '\n')
            if include_cell:
                fh.write(BEGIN_CELL+'\n'+'\n'.join(self._cell_string)+'\n'+END_CELL + '\n')
            
            for ii, shot in self._shots.iterrows():
                fh.write(BEGIN_CHUNK + '\n')
                fh.write(f'Image filename: {shot.file}\n')
                fh.write(f'Event: {shot.Event}\n')
                fh.write(f'Image serial number: {shot.serial - self.serial_offset}\n')
                keys = set(shot.keys()).difference(
                    {'Event', 'file', 'serial', 'shot_in_subset', 'subset'})
                for k in keys:
                    fh.write(f'{k} = {shot[k]}\n')
                if include_peaks:
                    fh.write(BEGIN_PEAKS + '\n')
                    self._peaks.loc[self._peaks.serial==shot.serial, PEAK_COLUMNS].to_csv(
                        fh, sep=' ', index=False, na_rep='-nan')
                    fh.write(END_PEAKS + '\n')

                crystals = self._crystals.loc[self._crystals.serial==shot.serial,:]

                for cid, crs in crystals.iterrows():
                    fh.write(BEGIN_CRYSTAL + '\n')
                    fh.write(f'Cell parameters {crs.a} {crs.b} {crs.c} nm, {crs.al} {crs.be} {crs.ga} deg\n')
                    fh.write(f'astar = {crs.astar_x} {crs.astar_y} {crs.astar_z} nm^-1\n')
                    fh.write(f'bstar = {crs.bstar_x} {crs.bstar_y} {crs.bstar_z} nm^-1\n')
                    fh.write(f'cstar = {crs.cstar_x} {crs.cstar_y} {crs.cstar_z} nm^-1\n')
                    fh.write(f'diffraction_resolution_limit = {crs.diff_limit} nm^-1 or {10/crs.diff_limit} A\n')
                    fh.write(f'predict_refine/det_shift x = {crs.xshift} y = {crs.yshift} mm\n')
                    keys = set(crs.keys()).difference(
                        {'Event', 'file', 'serial', 'shot_in_subset', 'subset',
                        'a', 'b', 'c', 'al', 'be', 'ga', 
                        'astar_x', 'astar_y', 'astar_z',
                        'bstar_x', 'bstar_y', 'bstar_z',
                        'cstar_x', 'cstar_y', 'cstar_z',
                        'diff_limit', 'xshift', 'yshift'})    
                    for k in keys:
                        fh.write(f'{k} = {crs[k]}\n') 
                    if include_indexed:
                        fh.write(BEGIN_REFLECTIONS + '\n')
                        self._indexed.loc[self._indexed.serial==shot.serial, REFLECTION_COLUMNS].to_csv(
                            fh, sep=' ', index=False, na_rep='-nan')
                        fh.write(END_REFLECTIONS + '\n')
                    fh.write(END_CRYSTAL + '\n')                    
                fh.write(END_CHUNK + '\n')


    def change_path(self, new_folder=None, old_pattern=None, new_pattern=None):
        
        for df in [self._crystals, self._shots, self._indexed, self._peaks]:
            if (new_folder is not None) and (old_pattern is not None):
                df.file = new_folder + '/' + \
                    df.file.str.rsplit('/', 1, True).iloc[:,-1].str.replace(old_pattern, new_pattern)
            elif old_pattern is not None:
                df.file = df.file.str.replace(old_pattern, new_pattern)
            elif new_folder is not None:
                df.file = new_folder + '/' + df.file.str.rsplit('/', 1, True).iloc[:,-1]


    def get_cxi_format(self, what='peaks', shots=None, half_pixel_shift=True):

        if shots is None:
            shots = self.shots

        if half_pixel_shift:
            off = -.5
        else:
            off = 0

        if what == 'peaks':
            ifield = 'Intensity'
            indexed = False
        elif what in ['indexed', 'predict', 'prediction']:
            ifield = 'I'
            indexed = True
        else:
            raise ValueError('what must be peaks or indexed')

        # some majig to get CXI arrays
        if indexed:
            self._indexed['pk_id'] = self._indexed.groupby(['file', 'Event']).cumcount()
            pk2 = self._indexed.set_index(['file', 'Event', 'pk_id'])
        else:
            self._peaks['pk_id'] = self._peaks.groupby(['file', 'Event']).cumcount()
            pk2 = self._peaks.set_index(['file', 'Event', 'pk_id'])
        # joining step with shot list is required to make sure that shots without peaks/indexing stay in
        s2 = shots[['file', 'Event']].set_index(['file', 'Event'])
        s2.columns = pd.MultiIndex.from_arrays([[], []], names=('field', 'pk_id'))
        pk2 = s2.join(pk2.unstack(-1), how='left')
        if indexed:
            self._indexed.drop('pk_id', axis=1)
        else:
            self._peaks.drop('pk_id', axis=1)

        cxidat = {
            'peakXPosRaw': (pk2['fs/px'] + off).fillna(0).values,
            'peakYPosRaw': (pk2['ss/px'] + off).fillna(0).values,
            'peakTotalIntensity': pk2[ifield].fillna(0).values,
            'nPeaks': pk2['fs/px'].notna().sum(axis=1).values}

        if indexed:
            cxidat.update({'peakSNR': (pk2[ifield]/pk2['Sigma(I)']).fillna(0).values,
                           'indexH': pk2['h'].fillna(0).values.astype(np.int),
                           'indexK': pk2['k'].fillna(0).values.astype(np.int),
                           'indexL': pk2['l'].fillna(0).values.astype(np.int)})

        return cxidat
