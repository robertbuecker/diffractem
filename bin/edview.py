#!/usr/bin/env python
import hdf5plugin
from diffractem.stream_parser import StreamParser
from diffractem.dataset import Dataset
import argparse
import pandas as pd
import numpy as np
import h5py
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import (QPushButton, QSpinBox, QCheckBox,
                             QTextEdit, QWidget, QApplication, QGridLayout, QTableWidget, QTableWidgetItem)
from diffractem.adxv import Adxv
from warnings import warn
from typing import Optional, Union
from time import sleep

# non-trivial detector geometries are currently not supported (licensing trouble)
# from cfelpyutils.crystfel_utils import load_crystfel_geometry
# from cfelpyutils.geometry_utils import apply_geometry_to_data, compute_visualization_pix_maps

pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp()

class EDViewer(QWidget):

    def __init__(self, args):

        super().__init__()
        self.dataset = Dataset()
        self.args = args
        self.data_path = None
        self.current_shot = pd.Series()
        self.diff_image = np.empty((0,0))
        self.map_image = np.empty((0,0))
        self.init_widgets()
        self.adxv = None
        self.geom = None

        self.read_files()
        self.switch_shot(0)

        if self.args.internal:
            self.hist_img.setLevels(np.quantile(self.diff_image, 0.02), np.quantile(self.diff_image, 0.98))

        self.update()

        self.show()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self.args.internal:
            self.adxv.exit()
        a0.accept()

    def read_files(self):

        file_type = args.filename.rsplit('.', 1)[-1]

        if file_type == 'stream':
            print(f'Parsing stream file {args.filename}...')
            stream = StreamParser(args.filename)
            # with open('tmp.geom', 'w') as fh:
            #     fh.write('\n'.join(stream._geometry_string))
            # self.geom = load_crystfel_geometry('tmp.geom')
            # os.remove('tmp.geom')
            # if len(self.geom['panels']) == 1:
            #     print('Single-panel geometry, so ignoring transforms for now.')
            #     #TODO make this more elegant, e.g. by overwriting image transform func with identity
            #     self.geom = None
            self.geom = None
            
            try:
                self.data_path = stream.geometry['data']
            except KeyError:
                if args.geometry is None:
                    raise ValueError('No data location specified in geometry file. Please use -d parameter.')

            files = sorted(list(stream.shots['file'].unique()))
            # print('Loading data files found in stream... \n', '\n'.join(files))
            try:
                self.dataset = Dataset.from_files(files, load_tables=False, init_stacks=False, open_stacks=False)
                self.dataset.load_tables(features=True)
                # print(self.dataset.shots.columns)
                self.dataset.merge_stream(stream)
                # get_selection would not be the right method to call (changes IDs), instead do...
                self.dataset._shots = self.dataset._shots.loc[self.dataset._shots.selected,:].reset_index(drop=True)
                # TODO get subset for incomplete coverage
                print('Merged stream and hdf5 shot lists')
            except Exception as err:
                self.dataset = Dataset()
                self.dataset._shots = stream.shots
                self.dataset._peaks = stream.peaks
                self.dataset._predict = stream.indexed
                self.dataset._shots['selected'] = True
                print('Could not load shot lists from H5 files, but have that from the stream file.')
                print(f'Reason: {err}')

        if args.geometry is not None:
            raise ValueError('Geometry files are currently not supported.')
            # self.geom = load_crystfel_geometry(args.geometry)

        if file_type in ['lst', 'h5', 'hdf', 'nxs']:
            self.dataset = Dataset.from_list(args.filename, load_tables=True, init_stacks=False, open_stacks=False)
            if not self.dataset.shots.selected.all():
                # dirty removal of unwanted shots is sufficient in this case:
                self.dataset._shots = self.dataset._shots.loc[self.dataset._shots.selected,:].reset_index(drop=True)

        if args.data_path is not None:
            self.data_path = args.data_path

        if self.data_path is None:
            # data path neither set via stream file, nor explicitly. We have to guess.
            try:
                with h5py.File(self.dataset.shots.file.iloc[0], 'r') as fh:
                    base = '/%/data'.replace('%', self.dataset.shots.subset.iloc[0])
                    self.data_path = '/%/data/' + fh[base].attrs['signal']
                print('Found data path', self.data_path)
            except Exception as err:
                warn(str(err), RuntimeWarning)
                print('Could not find out data path. Assuming /%/data/raw_counts')
                self.data_path = '/%/data/raw_counts'

        if self.args.query:
            print('Only showing shots with', self.args.query)
            #self.dataset.select(self.args.query)
            #self.dataset = self.dataset.get_selection(self.args.query, file_suffix=None, reset_id=False)
            #print('cutting shot list only')
            self.dataset._shots = self.dataset._shots.query(args.query)

        if self.args.sort_crystals:
            print('Re-sorting shots by region/crystal/run.')
            self.dataset._shots = self.dataset._shots.sort_values(by=['sample', 'region', 'crystal_id', 'run'])

        if not self.args.internal:
            #adxv_args = {'wavelength': 0.0251, 'distance': 2280, 'pixelsize': 0.055}
            adxv_args = {}
            self.adxv = Adxv(hdf5_path=self.data_path.replace('%', 'entry'),
                             adxv_bin=self.args.adxv_bin, **adxv_args)

        self.b_goto.setMaximum(self.dataset.shots.shape[0]-1)
        self.b_goto.setMinimum(0)

    def update_image(self):
        print(self.current_shot)
        with h5py.File(self.current_shot['file'], mode='r') as f:

            if self.args.internal:
                path = self.data_path.replace('%', self.current_shot.subset)
                print('Loading {}:{} from {}'.format(path,
                                                     self.current_shot['shot_in_subset'], self.current_shot['file']))                
                if len(f[path].shape) == 3:
                    self.diff_image = f[path][int(self.current_shot['shot_in_subset']), ...]
                elif len(f[path].shape) == 2:
                    self.diff_image = f[path][:]
                    
                self.diff_image[np.isnan(self.diff_image)] = 0
                self.hist_img.setHistogramRange(np.partition(self.diff_image.flatten(), 100)[100], np.partition(self.diff_image.flatten(), -100)[-100])

                levels = self.hist_img.getLevels()
                # levels = (max(levels[0], -1), levels[1])
                levels = (levels[0], levels[1])
                if self.geom is not None:
                    raise RuntimeError('This should not happen')
                    # self.diff_image = apply_geometry_to_data(self.diff_image, self.geom)
                self.img.setImage(self.diff_image, autoRange=False)
                
                self.img.setLevels(levels)
                self.hist_img.setLevels(levels[0], levels[1])

            if not self.args.no_map:
                try:
                    path = args.map_path.replace('%', self.current_shot['subset'])
                    self.map_image = f[path][...]
                    self.mapimg.setImage(self.map_image)
                except KeyError:
                     warn('No map found at {}!'.format(path), Warning)

        if not self.args.internal:
            self.adxv.load_image(self.current_shot.file)
            self.adxv.slab(self.current_shot.shot_in_subset + 1)

    def update_plot(self):

        allpk = []

        if self.b_peaks.isChecked():
            
            if (len(self.dataset.peaks) == 0) or args.cxi_peaks:
                path = args.cxi_peaks_path.replace('%', self.current_shot.subset)
                print('Loading CXI peaks of {}:{} from {}'.format(path,
                                                        self.current_shot['shot_in_subset'], self.current_shot['file']))     
                with h5py.File(self.current_shot.file) as fh:
                    ii = int(self.current_shot['shot_in_subset'])
                    Npk = fh[path + '/nPeaks'][ii]
                    x = fh[path + '/peakXPosRaw'][ii, :Npk]
                    y = fh[path + '/peakYPosRaw'][ii, :Npk]

                peaks = pd.DataFrame((x, y), index=['fs/px', 'ss/px']).T

            else:
                peaks = self.dataset.peaks.loc[(self.dataset.peaks.file == self.current_shot.file)
                                           & (self.dataset.peaks.Event == self.current_shot.Event),
                                           ['fs/px', 'ss/px']] - 0.5
                x = peaks.loc[:,'fs/px']
                y = peaks.loc[:,'ss/px']

            if self.geom is not None:
                raise RuntimeError('Someone set geom to something. This should not happen.')
                # print('Projecting peaks...')
                # maps = compute_visualization_pix_maps(self.geom)
                # x = maps.x[y.astype(int), x.astype(int)]
                # y = maps.y[y.astype(int), x.astype(int)]

            if self.args.internal:
                ring_pen = pg.mkPen('g', width=0.8)
                self.found_peak_canvas.setData(x, y,
                symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), antialias=True)
            else:
                allpk.append(peaks.assign(group=0))

        else:
            self.found_peak_canvas.clear()

        if self.b_pred.isChecked() and (self.dataset.predict.shape[0] > 0):
            
            pred = self.dataset.predict.loc[(self.dataset.predict.file == self.current_shot.file)
                                           & (self.dataset.predict.Event == self.current_shot.Event),
                                           ['fs/px', 'ss/px']] - 0.5

            if self.geom is not None:
                raise RuntimeError('Someone set geom to not None. This should not happen.')
                # print('Projecting predictions...')
                # maps = compute_visualization_pix_maps(self.geom)
                # x = maps.x[pred.loc[:,'ss/px'].astype(int),
                #     pred.loc[:,'fs/px'].astype(int)]
                # y = maps.y[pred.loc[:,'ss/px'].astype(int),
                #     pred.loc[:,'fs/px'].astype(int)]
            else:
                x = pred.loc[:,'fs/px']
                y = pred.loc[:,'ss/px']

            if self.args.internal:
                square_pen = pg.mkPen('r', width=0.8)
                self.predicted_peak_canvas.setData(x, y,
                                              symbol='s', size=13, pen=square_pen, brush=(0, 0, 0, 0), antialias=True)
            else:
                allpk.append(pred.assign(group=1))

        else:
            self.predicted_peak_canvas.clear()

        if not self.args.internal and len(allpk) > 0:
            self.adxv.define_spot('green', 5, 0, 0)
            self.adxv.define_spot('red', 0, 10, 1)
            self.adxv.load_spots(pd.concat(allpk, axis=0, ignore_index=True).values)
        elif not self.args.internal:
            self.adxv.load_spots(np.empty((0,3)))

        if self.dataset.features.shape[0] > 0:
            ring_pen = pg.mkPen('g', width=2)
            dot_pen = pg.mkPen('y', width=0.5)

            region_feat = self.dataset.features.loc[(self.dataset.features['region'] == self.current_shot['region'])
                                               & (self.dataset.features['sample'] == self.current_shot['sample'])]
            
            print('Number of region features:', region_feat.shape[0])

            if self.current_shot['crystal_id'] != -1:
                single_feat = region_feat.loc[region_feat['crystal_id'] == self.current_shot['crystal_id'], :]
                x0 = single_feat['crystal_x'].squeeze()
                y0 = single_feat['crystal_y'].squeeze()
                self.found_features_canvas.setData(region_feat['crystal_x'], region_feat['crystal_y'],
                                              symbol='+', size=7, pen=dot_pen, brush=(0, 0, 0, 0), pxMode=True)

                if self.b_zoom.isChecked():
                    self.map_box.setRange(xRange=(x0 - 5 * args.beam_diam, x0 + 5 * args.beam_diam),
                                     yRange=(y0 - 5 * args.beam_diam, y0 + 5 * args.beam_diam))
                    self.single_feature_canvas.setData([x0], [y0],
                                                  symbol='o', size=args.beam_diam, pen=ring_pen,
                                                  brush=(0, 0, 0, 0), pxMode=False)
                    try:
                        c_real = np.cross([self.current_shot.astar_x, self.current_shot.astar_y, self.current_shot.astar_z],
                                          [self.current_shot.bstar_x, self.current_shot.bstar_y, self.current_shot.bstar_z])
                        b_real = np.cross([self.current_shot.cstar_x, self.current_shot.cstar_y, self.current_shot.cstar_z],
                                          [self.current_shot.astar_x, self.current_shot.astar_y, self.current_shot.astar_z])
                        a_real = np.cross([self.current_shot.bstar_x, self.current_shot.bstar_y, self.current_shot.bstar_z],
                                          [self.current_shot.cstar_x, self.current_shot.cstar_y, self.current_shot.cstar_z])
                        a_real = 20 * a_real / np.sum(a_real ** 2) ** .5
                        b_real = 20 * b_real / np.sum(b_real ** 2) ** .5
                        c_real = 20 * c_real / np.sum(c_real ** 2) ** .5
                        self.a_dir.setData(x=x0 + np.array([0, a_real[0]]), y=y0 + np.array([0, a_real[1]]))
                        self.b_dir.setData(x=x0 + np.array([0, b_real[0]]), y=y0 + np.array([0, b_real[1]]))
                        self.c_dir.setData(x=x0 + np.array([0, c_real[0]]), y=y0 + np.array([0, c_real[1]]))
                    except:
                        print('Could not read lattice vectors.')
                else:
                    self.single_feature_canvas.setData([x0], [y0],
                                                  symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), pxMode=True)
                    self.map_box.setRange(xRange=(0, self.map_image.shape[1]), yRange=(0, self.map_image.shape[0]))



            else:
                self.single_feature_canvas.setData([], [])

    def update(self):

        self.found_peak_canvas.clear()
        self.predicted_peak_canvas.clear()
        app.processEvents()

        self.update_image()   
        if args.cxi_peaks and not args.internal:
            # give adxv some time to display the image before accessing the CXI data
            sleep(0.2)
        self.update_plot()

        print(self.current_shot)

    # CALLBACK FUNCTIONS

    def switch_shot(self, shot_id=None):
        if shot_id is None:
            shot_id = self.b_goto.value()

        self.shot_id = max(0, shot_id % self.dataset.shots.shape[0])
        self.current_shot = self.dataset.shots.iloc[self.shot_id, :]
        self.meta_table.setRowCount(self.current_shot.shape[0])
        self.meta_table.setColumnCount(2)

        for row, (k, v) in enumerate(self.current_shot.items()):
            self.meta_table.setItem(row, 0, QTableWidgetItem(k))
            self.meta_table.setItem(row, 1, QTableWidgetItem(str(v)))

        self.meta_table.resizeRowsToContents()

        shot = self.current_shot
        title = {'sample': '', 'region': 'Reg', 'feature': 'Feat', 'frame': 'Frame', 'event': 'Ev', 'file': ''}
        titlestr = ''
        for k, v in title.items():
            titlestr += f'{v} {shot[k]}' if k in shot.keys() else ''
        titlestr += f' ({shot.name} of {self.dataset.shots.shape[0]})'
        print(titlestr)

        self.setWindowTitle(titlestr)

        self.b_goto.blockSignals(True)
        self.b_goto.setValue(self.shot_id)
        self.b_goto.blockSignals(False)

        self.update()

    def switch_shot_rel(self, shift):
        self.switch_shot(self.shot_id + shift)

    def mouse_moved(self, evt):
        mousePoint = self.img.mapFromDevice(evt[0])
        x, y = round(mousePoint.x()), round(mousePoint.y())
        x = min(max(0, x), self.diff_image.shape[1] - 1)
        y = min(max(0, y), self.diff_image.shape[0] - 1)
        I = self.diff_image[y, x]
        #print(x, y, I)
        self.info_text.setPos(x, y)
        self.info_text.setText(f'{x:0.1f}, {y:0.1f}: {I:0.1f}')

    def init_widgets(self):

        self.imageWidget = pg.GraphicsLayoutWidget()

        # IMAGE DISPLAY

        # A plot area (ViewBox + axes) for displaying the image
        self.image_box = self.imageWidget.addViewBox()
        self.image_box.setAspectLocked()

        self.img = pg.ImageItem()
        self.img.setZValue(0)
        self.image_box.addItem(self.img)
        self.proxy = pg.SignalProxy(self.img.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        self.found_peak_canvas = pg.ScatterPlotItem()
        self.image_box.addItem(self.found_peak_canvas)
        self.found_peak_canvas.setZValue(2)

        self.predicted_peak_canvas = pg.ScatterPlotItem()
        self.image_box.addItem(self.predicted_peak_canvas)
        self.predicted_peak_canvas.setZValue(2)

        self.info_text = pg.TextItem(text='')
        self.image_box.addItem(self.info_text)
        self.info_text.setPos(0, 0)

        # Contrast/color control
        self.hist_img = pg.HistogramLUTItem(self.img, fillHistogram=False)
        self.imageWidget.addItem(self.hist_img)

        # MAP DISPLAY

        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_widget.setWindowTitle('region map')

        # Map image control
        self.map_box = self.map_widget.addViewBox()
        self.map_box.setAspectLocked()

        self.mapimg = pg.ImageItem()
        self.mapimg.setZValue(0)
        self.map_box.addItem(self.mapimg)

        self.found_features_canvas = pg.ScatterPlotItem()
        self.map_box.addItem(self.found_features_canvas)
        self.found_features_canvas.setZValue(2)

        self.single_feature_canvas = pg.ScatterPlotItem()
        self.map_box.addItem(self.single_feature_canvas)
        self.single_feature_canvas.setZValue(2)

        # lattice vectors
        self.a_dir = pg.PlotDataItem(pen=pg.mkPen('r', width=1))
        self.b_dir = pg.PlotDataItem(pen=pg.mkPen('g', width=1))
        self.c_dir = pg.PlotDataItem(pen=pg.mkPen('b', width=1))
        self.map_box.addItem(self.a_dir)
        self.map_box.addItem(self.b_dir)
        self.map_box.addItem(self.c_dir)

        # Contrast/color control
        self.hist_map = pg.HistogramLUTItem(self.mapimg)
        self.map_widget.addItem(self.hist_map)

        ### CONTROl BUTTONS

        b_rand = QPushButton('rnd')
        b_plus10 = QPushButton('+10')
        b_minus10 = QPushButton('-10')
        b_last = QPushButton('last')
        self.b_peaks = QCheckBox('peaks')
        self.b_pred = QCheckBox('crystal')
        self.b_zoom = QCheckBox('zoom')
        b_reload = QPushButton('reload')
        self.b_goto = QSpinBox()

        b_rand.clicked.connect(lambda: self.switch_shot(np.random.randint(0, self.dataset.shots.shape[0] - 1)))
        b_plus10.clicked.connect(lambda: self.switch_shot_rel(+10))
        b_minus10.clicked.connect(lambda: self.switch_shot_rel(-10))
        b_last.clicked.connect(lambda: self.switch_shot(self.dataset.shots.index.max()))
        self.b_peaks.stateChanged.connect(self.update)
        self.b_pred.stateChanged.connect(self.update)
        self.b_zoom.stateChanged.connect(self.update)
        b_reload.clicked.connect(lambda: self.read_files())
        self.b_goto.valueChanged.connect(lambda: self.switch_shot(None))

        self.button_layout = QtGui.QGridLayout()
        self.button_layout.addWidget(b_plus10, 0, 2)
        self.button_layout.addWidget(b_minus10, 0, 1)
        self.button_layout.addWidget(b_rand, 0, 4)
        self.button_layout.addWidget(b_last, 0, 3)
        self.button_layout.addWidget(self.b_goto, 0, 0)
        self.button_layout.addWidget(b_reload, 0, 10)
        self.button_layout.addWidget(self.b_peaks, 0, 21)
        self.button_layout.addWidget(self.b_pred, 0, 22)
        self.button_layout.addWidget(self.b_zoom, 0, 23)

        self.meta_table = QTableWidget()
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.horizontalHeader().setVisible(False)
        self.meta_table.setFont(QtGui.QFont('Helvetica', 10))

        # --- TOP-LEVEL ARRANGEMENT
        self.top_layout = QGridLayout()
        self.setLayout(self.top_layout)

        if self.args.internal:
            self.top_layout.addWidget(self.imageWidget, 0, 0)
            self.top_layout.setColumnStretch(0, 2)
            
        if not self.args.no_map:
            self.top_layout.addWidget(self.map_widget, 0, 1)
            self.top_layout.setColumnStretch(1, 1.5)
            
        self.top_layout.addWidget(self.meta_table, 0, 2)
        self.top_layout.addLayout(self.button_layout, 1, 0, 1, 3)
        
        self.top_layout.setColumnStretch(2, 0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Viewer for Serial Electron Diffraction data')
    parser.add_argument('filename', type=str, help='Stream file, list file, or HDF5')
    parser.add_argument('-g', '--geometry', type=str, help='CrystFEL geometry file, might be helpful')
    parser.add_argument('-q', '--query', type=str, help='Query string to filter shots by column values')
    parser.add_argument('-d', '--data_path', type=str, help='Data field in HDF5 file(s). Defaults to stream file or tries a few.')
    parser.add_argument('--internal', help='Use internal diffraction viewer instead of adxv', action='store_true')
    parser.add_argument('--adxv-bin', help='Location/command string of adxv binary', default='adxv')
    parser.add_argument('--map-path', type=str, help='Path to map image', default='/%/map/image')
    parser.add_argument('--feature-path', type=str, help='Path to map feature table', default='/%/map/features')
    parser.add_argument('--cxi-peaks', help='Prefer CXI-format peaks in HDF5 files over stream/HDF5 table', action='store_true')
    parser.add_argument('--cxi-peaks-path', type=str, help='Path to CXI peaks table', default='/%/data')
    parser.add_argument('--peaks-path', type=str, help='Path to peaks table in HDF5 files', default='/%/results/peaks')
    parser.add_argument('--predict-path', type=str, help='Path to prediction table', default='/%/results/predict')
    parser.add_argument('--no-map', help='Hide map, even if we had it', action='store_true')
    parser.add_argument('--beam-diam', type=int, help='Beam size displayed in real space, in pixels', default=5)
    parser.add_argument('--sort-crystals', help='Sort shots by crystal IDs', action='store_true')

    args = parser.parse_args()

    # operation modes:
    # (1) file list (+ geometry) + nxs: estimate geometry from nxs if geometry is absent
    # (2) expanded file list (+ geometry) + nxs: first match nxs self.current_shot lists vs expanded file list
    # (3) (expanded) file list + geometry + hdf5: omit map image automatically
    # (4) stream + nxs: as (2), peaks/predict in stream take precedence over nxs
    # (5) stream + hdf5: as (3)

    # TODO next: work on read_file
    viewer = EDViewer(args)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app.instance().exec_()
