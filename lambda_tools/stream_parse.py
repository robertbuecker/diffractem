import pandas as pd
from io import StringIO

class StreamParser:

    def __init__(self, filename, serial_offset=-1):

        self.geometry = {}
        self.cell = {}
        self.merge_shot = {}

        linedat_peak = []
        linedat_index = []
        shotlist = []
        init_peak = False
        init_index = False
        init_geom = False

        with open(filename, 'r') as fh:
            fstr = StringIO(fh.read())

        for ln, l in enumerate(fstr):

            # Geometry file
            if 'End geometry file' in l:
                init_geom = False
            elif init_geom and ('=' in l):
                k, v = l.split('=', 1)
                try:
                    self.geometry[k.strip()] = float(v)
                except ValueError:
                    self.geometry[k.strip()] = v.strip()
            elif 'Begin geometry file' in l:
                init_geom = True

            # Cell file
            if 'End unit cell' in l:
                init_geom = False
            elif init_geom and ('=' in l):
                k, v = l.split('=', 1)
                try:
                    self.cell[k.strip()] = float(v)
                except ValueError:
                    self.cell[k.strip()] = v.strip()
            elif 'Begin unit cell' in l:
                init_geom = True

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
                crystal_info.pop('dummy')
                shotdat.update(crystal_info)
            elif 'Cell parameters' in l:
                crystal_info.update(
                    {k: v for k, v in zip(['a', 'b', 'c', 'dummy', 'al', 'be', 'ga'], l.split(' ')[2:9])})
            elif 'astar' in l:
                crystal_info.update({k: v for k, v in zip(['astar_x', 'astar_y', 'astar_z'], l.split(' ')[2:5])})
            elif 'bstar' in l:
                crystal_info.update({k: v for k, v in zip(['bstar_x', 'bstar_y', 'bstar_z'], l.split(' ')[2:5])})
            elif 'cstar' in l:
                crystal_info.update({k: v for k, v in zip(['cstar_x', 'cstar_y', 'cstar_z'], l.split(' ')[2:5])})
            elif 'diffraction_resolution_limit' in l:
                crystal_info['diff_limit'] = l.rsplit(' nm', 1)[0].rsplit('= ', 1)[-1]
            elif 'predict_refine/det_shift' in l:
                crystal_info['xshift'] = l.split(' ')[3]
                crystal_info['yshift'] = l.split(' ')[6]
                continue
            elif 'num_implausible_reflections' in l:
                crystal_info['implausible'] = l.rsplit(' ')[-1]

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
    def indexed(self):
        return self._indexed

    @property
    def peaks(self):
        return self._peaks

    @property
    def shots(self):
        return self._peaks
