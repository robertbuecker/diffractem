import pandas as pd
from io import StringIO

class StreamParser:

    def __init__(self, filename, serial_offset=-1):

        self.merge_shot = {}
        self.command = ''
        self._cell_string = []
        self._geometry_string = []

        linedat_peak = []
        linedat_index = []
        shotlist = []
        init_peak = False
        init_index = False
        init_geom = False
        init_cell = False

        with open(filename, 'r') as fh:
            fstr = StringIO(fh.read())

        for ln, l in enumerate(fstr):

            # Call string
            if 'indexamajig' in l:
                self.command = l

            # Geometry file
            elif 'End geometry file' in l:
                init_geom = False
                self._geometry_string.append(l.strip())
            elif init_geom and ('=' in l):
                self._geometry_string.append(l.strip())
            elif 'Begin geometry file' in l:
                init_geom = True
                self._geometry_string.append(l.strip())

            # Cell file
            elif 'End unit cell' in l:
                init_cell = False
                self._cell_string.append(l.strip())
            elif init_cell and ('=' in l):
                self._cell_string.append(l.strip())
            elif 'Begin unit cell' in l:
                init_cell = True
                self._cell_string.append(l.strip())

            # Event chunks
            elif 'Begin chunk' in l:
                shotdat = {'Event': -1, 'shot_in_subset': -1, 'subset': '',
                           'file': '', 'serial': -1, 'indexer': '(none)'}
            elif 'End chunk' in l:
                shotlist.append(shotdat)
            # Event descriptors, indexing scheme
            elif 'Event:' in l:
                shotdat['Event'] = l.split(': ')[-1].strip()
                shotdat['shot_in_subset'] = int(shotdat['Event'].split('//')[-1])
                shotdat['subset'] = shotdat['Event'].split('//')[0].strip()
            elif 'Image filename:' in l:
                shotdat['file'] = l.split(':')[-1].strip()
            elif 'Image serial number:' in l:
                shotdat['serial'] = int(l.split(': ')[1]) + serial_offset
            elif 'indexed_by' in l:
                shotdat['indexer'] = l.split(' ')[2].replace("\n", "")

            # Actual parsing (found peaks)
            elif 'End of peak list' in l:
                init_peak = False
            elif init_peak:
                linedat_peak.append(
                    '{} {} {} {}'.format(l.strip(), shotdat['file'], shotdat['Event'], shotdat['serial']))
            elif 'fs/px   ss/px (1/d)/nm^-1   Intensity  Panel' in l:
                init_peak = True

            # Information from indexing
            elif 'Begin crystal' in l:
                crystal_info = {}
            elif 'End crystal' in l:
                shotdat.update(crystal_info)
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
            elif 'num_implausible_reflections' in l:
                crystal_info['implausible'] = int(l.rsplit(' ')[-1])

            # Actual parsing (indexed peaks)
            elif 'End of reflections' in l:
                init_index = False
            elif init_index:
                linedat_index.append(
                    '{} {} {} {}'.format(l.strip(), shotdat['file'], shotdat['Event'], shotdat['serial']))
            elif 'h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel' in l:
                init_index = True

        self._peaks = pd.read_csv(StringIO('\n'.join(linedat_peak)), delim_whitespace=True, header=None,
                              names=['fs/px', 'ss/px', '(1/d)/nm^-1', 'Intensity', 'Panel', 'file', 'Event', 'serial']
                              ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(
            drop=True).drop('index', axis=1)

        self._indexed = pd.read_csv(StringIO('\n'.join(linedat_index)), delim_whitespace=True, header=None,
                               names=['h', 'k', 'l', 'I', 'Sigma(I)', 'Peak', 'Background', 'fs/px', 'ss/px', 'Panel',
                                      'file', 'Event', 'serial']
                               ).sort_values('serial').reset_index().sort_values(['serial', 'index']).reset_index(
            drop=True).drop('index', axis=1)

        self._shots = pd.DataFrame(shotlist)

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
            try:
                g[k.strip()] = float(v)
            except ValueError:
                g[k.strip()] = v.strip()

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
        for opt in self.command.split('--')[1:]:
            if '=' in opt:
                k, v = opt.split('=', 1)
                try:
                    o[k.strip()] = float(v)
                except ValueError:
                    o[k.strip()] = v.strip()

        return o

    @property
    def indexed(self):
        return self._indexed

    @property
    def peaks(self):
        return self._peaks

    @property
    def shots(self):
        return self._peaks
