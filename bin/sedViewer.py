import hdf5plugin
from diffractem.io import *
from diffractem.nexus import get_table
from diffractem.stream_parser import StreamParser
from diffractem.dataset import Dataset
import pyqtgraph as pg
import importlib
import argparse

PyQt4_found = importlib.util.find_spec("PyQt4")
if PyQt4_found is not None:
    from PyQt4 import QtGui, QtCore
else:
    from PyQt5 import QtGui, QtCore

pg.setConfigOptions(imageAxisOrder='row-major')

def read_files():
    global args, data_path, dataset

    # Mandatory stuff: data path, shot list

    data_path = None
    files = []
    file_type = args.filename.rsplit('.', 1)[-1]

    if file_type == 'stream':
        print(f'Parsing stream file {args.filename}...')
        stream = StreamParser(args.filename)
        data_path = stream.geometry['data']
        files = list(stream.shots['file'].unique())
        try:
            dataset = Dataset.from_list(files, load_tables=False)
            dataset.load_tables(features=True)
            dataset.merge_stream(stream)
            # TODO get subset for incomplete coverage
            print('Merged stream and hdf5 shot lists')
        except Exception as err:
            dataset = Dataset()
            dataset.merge_stream(stream)
            print('Could not load shot lists from H5 files, but have that from the stream file.')
            print(f'Reason: {err}')

    if file_type in ['lst', 'h5', 'hdf', 'nxs']:
        dataset = Dataset.from_list(args.filename, load_tables=True)

    if args.data_path is not None:
        data_path = args.data_path

    if data_path is None:
        # data path neither set via stream file, nor explicitly. We have to guess.
        data_path = ['/%/data/centered_fr', '/%/data/centered', '/%/data/masked', '/%/data/raw_counts']

    if args.geometry is not None:
        raise NotImplementedError('Explicit geometry files are not allowed yet. Sry.')

    if args.query:
        nshots0 = dataset.shots.shape[0]
        dataset = dataset.get_selection(args.query)
        nshots = dataset.shots.shape[0]
        print(f'{nshots} shots from {nshots0} selected by query f{args.query}')

# CALLBACK FUNCTIONS

def switch_shot(serial):
    global imageSerialNumber, shot, dataset
    imageSerialNumber = max(0, serial % dataset.shots.shape[0])
    shot = dataset.shots.iloc[imageSerialNumber, :]
    update()


def switch_shot_rel(shift):
    global imageSerialNumber
    switch_shot(imageSerialNumber + shift)


def toggleMrkerButton_clicked():
    global show_markers
    show_markers = not show_markers
    update()


def toggleFoundPeaksButton_clicked():
    global show_peaks
    show_peaks = not show_peaks
    update()


def toggleFoundCrystalButton_clicked():
    global show_predict
    show_predict = not show_predict
    update()


def zoomOnCrystalButton_clicked():
    global map_zoomed
    map_zoomed = not map_zoomed
    update()


def updateImage():
    global imageSerialNumber, rawImage, mapImage, data_path, shot

    try:
        if isinstance(data_path, str):
            data_path = [data_path]
        with h5py.File(shot['file'], mode='r', swmr=True) as f:
            for img_array in data_path:
                try:
                    path = img_array.replace('%', shot['subset'])
                    rawImage = f[path][int(shot['shot_in_subset']), ...]
                    print('Loading {}:{} from {}'.format(path, shot['shot_in_subset'], shot['file']))
                    break
                except KeyError:
                    continue
            else:
                raise KeyError('None of the stack names {} found'.format(data_path))

            if show_map:
                mapImage = f[args.map_path.replace('%', shot['subset'])][...]

    except Exception as err:
        print('Could not load image data due to {}'.format(err))
        rawImage = rawImage

def updatePlot():
    global img, mapimg, hist_img, imageSerialNumber, rawImage, mapImage, map_zoomed, dataset

    if show_peaks and (dataset.peaks.shape[0] > 0) and show_markers:
        ring_pen = pg.mkPen('g', width=0.8)
        peaks = dataset.peaks.loc[(dataset.peaks.file == shot.file) & (dataset.peaks.Event == shot.Event), :]
        found_peak_canvas.setData(peaks['fs/px'] + 0.5, peaks['ss/px'] + 0.5,
                                  symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), antialias=True)

    else:
        found_peak_canvas.clear()

    if show_predict and (dataset.predict.shape[0] > 0) and show_markers:
        square_pen = pg.mkPen('r', width=0.8)
        predict = dataset.predict.loc[(dataset.predict.file == shot.file) & (dataset.predict.Event == shot.Event), :]
        predicted_peak_canvas.setData(predict['fs/px'] + 0.5, predict['ss/px'] + 0.5,
                                      symbol='s', size=13, pen=square_pen, brush=(0, 0, 0, 0), antialias=True)

    else:
        predicted_peak_canvas.clear()

    if dataset.features.shape[0] > 0:
        ring_pen = pg.mkPen('g', width=2)
        dot_pen = pg.mkPen('y', width=0.5)

        region_feat = dataset.features.loc[(dataset.features['region'] == shot['region'])
                                           & (dataset.features['sample'] == shot['sample'])
                                           & (dataset.features['run'] == shot['run']), :]

        if shot['crystal_id'] != -1:
            single_feat = region_feat.loc[region_feat['crystal_id'] == shot['crystal_id'], :]
            x0 = single_feat['crystal_x'].squeeze()
            y0 = single_feat['crystal_y'].squeeze()
            found_features_canvas.setData(region_feat['crystal_x'], region_feat['crystal_y'],
                                          symbol='+', size=7, pen=dot_pen, brush=(0, 0, 0, 0), pxMode=True)

            if map_zoomed:
                map_box.setRange(xRange=(x0 - 5 * args.beam_diam, x0 + 5 * args.beam_diam),
                                 yRange=(y0 - 5*args.beam_diam, y0 + 5*args.beam_diam))
                single_feature_canvas.setData([x0], [y0],
                                              symbol='o', size=args.beam_diam, pen=ring_pen,
                                              brush=(0, 0, 0, 0), pxMode=False)
                try:
                    c_real = np.cross([shot.astar_x, shot.astar_y, shot.astar_z],
                                      [shot.bstar_x, shot.bstar_y, shot.bstar_z])
                    b_real = np.cross([shot.cstar_x, shot.cstar_y, shot.cstar_z],
                                      [shot.astar_x, shot.astar_y, shot.astar_z])
                    a_real = np.cross([shot.bstar_x, shot.bstar_y, shot.bstar_z],
                                      [shot.cstar_x, shot.cstar_y, shot.cstar_z])
                    a_real = 20*a_real/np.sum(a_real**2)**.5
                    b_real = 20*b_real / np.sum(b_real ** 2) ** .5
                    c_real = 20*c_real / np.sum(c_real ** 2) ** .5
                    a_dir.setData(x=x0 + np.array([0, a_real[0]]), y=y0 + np.array([0, a_real[1]]))
                    b_dir.setData(x=x0 + np.array([0, b_real[0]]), y=y0 + np.array([0, b_real[1]]))
                    c_dir.setData(x=x0 + np.array([0, c_real[0]]), y=y0 + np.array([0, c_real[1]]))
                except:
                    print('Could not read lattice vectors.')
            else:
                single_feature_canvas.setData([x0], [y0],
                                              symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), pxMode=True)
                map_box.setRange(xRange=(0, mapImage.shape[1]), yRange=(0, mapImage.shape[0]))



        else:
            single_feature_canvas.setData([],[])

    levels = hist_img.getLevels()
    img.setImage(rawImage, autoRange=False)
    img.setLevels(levels)
    mapimg.setImage(mapImage)
    hist_img.setLevels(levels[0], levels[1])


def update():
    global imageSerialNumber, shot

    found_peak_canvas.clear()
    predicted_peak_canvas.clear()
    pg.QtGui.QApplication.processEvents()

    updateImage()
    updatePlot()

    print(shot)

    topWidget.setWindowTitle('{} Reg {} Run {} Feat {} Frame {} ({}//{} in {}, {} out of {}) '.format(shot['sample'],
                                                                                                      shot['region'],
                                                                                                      shot['run'],
                                                                                                      shot[
                                                                                                          'crystal_id'],
                                                                                                      shot['frame'],
                                                                                                      shot['subset'],
                                                                                                      shot[
                                                                                                          'shot_in_subset'],
                                                                                                      shot['file'],
                                                                                                      shot.name,
                                                                                                      dataset.shots.shape[
                                                                                                          0]))

def mouseMoved(evt):
    global rawImage
    mousePoint = img.mapFromDevice(evt[0])
    x, y = round(mousePoint.x()), round(mousePoint.y())
    x = min(max(0, x), rawImage.shape[1] - 1)
    y = min(max(0, y), rawImage.shape[0] - 1)
    I = rawImage[y, x]
    #print(x, y, I)
    info_text.setPos(x, y)
    info_text.setText('{}, {}: {}'.format(x, y, I))


# MAIN SETUP -------

pg.mkQApp()

imageWidget = pg.GraphicsLayoutWidget()
imageWidget.setWindowTitle('stream file viewer')

# IMAGE DISPLAY

# A plot area (ViewBox + axes) for displaying the image
image_box = imageWidget.addViewBox()
image_box.setAspectLocked()

img = pg.ImageItem()
img.setZValue(0)
image_box.addItem(img)
proxy = pg.SignalProxy(img.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

found_peak_canvas = pg.ScatterPlotItem()
image_box.addItem(found_peak_canvas)
found_peak_canvas.setZValue(2)

predicted_peak_canvas = pg.ScatterPlotItem()
image_box.addItem(predicted_peak_canvas)
predicted_peak_canvas.setZValue(2)

info_text = pg.TextItem(text='')
image_box.addItem(info_text)
info_text.setPos(0, 0)

# Contrast/color control
hist_img = pg.HistogramLUTItem(img)
imageWidget.addItem(hist_img)

# MAP DISPLAY

mapWidget = pg.GraphicsLayoutWidget()
mapWidget.setWindowTitle('region map')

# Map image control
map_box = mapWidget.addViewBox()
map_box.setAspectLocked()

mapimg = pg.ImageItem()
mapimg.setZValue(0)
map_box.addItem(mapimg)

found_features_canvas = pg.ScatterPlotItem()
map_box.addItem(found_features_canvas)
found_features_canvas.setZValue(2)

single_feature_canvas = pg.ScatterPlotItem()
map_box.addItem(single_feature_canvas)
single_feature_canvas.setZValue(2)

# lattice vectors
a_dir = pg.PlotDataItem(pen=pg.mkPen('r', width=1))
b_dir = pg.PlotDataItem(pen=pg.mkPen('g', width=1))
c_dir = pg.PlotDataItem(pen=pg.mkPen('b', width=1))
map_box.addItem(a_dir)
map_box.addItem(b_dir)
map_box.addItem(c_dir)

# Contrast/color control
hist_map = pg.HistogramLUTItem(mapimg)
mapWidget.addItem(hist_map)

# Control Buttons
nextImageButton = QtGui.QPushButton('+1')
previousImageButton = QtGui.QPushButton('-1')
randomImageButton = QtGui.QPushButton('rnd')
plus10ImageButton = QtGui.QPushButton('+10')
minus10ImageButton = QtGui.QPushButton('-10')
lastImageButton = QtGui.QPushButton('last')
toggleMarkerButton = QtGui.QPushButton('markers')
toggleFoundPeaksButton = QtGui.QPushButton('peaks')
toggleFoundCrystalButton = QtGui.QPushButton('crystal')
zoomOnCrystalButton = QtGui.QPushButton('zoom')
reloadButton = QtGui.QPushButton('reload')

nextImageButton.clicked.connect(lambda: switch_shot_rel(1))
previousImageButton.clicked.connect(lambda: switch_shot_rel(-1))
randomImageButton.clicked.connect(lambda: switch_shot(np.random.randint(0, dataset.shots.shape[0]-1)))
plus10ImageButton.clicked.connect(lambda: switch_shot_rel(+10))
minus10ImageButton.clicked.connect(lambda: switch_shot_rel(-10))
lastImageButton.clicked.connect(lambda: switch_shot(dataset.shots.index.max()))
toggleMarkerButton.clicked.connect(toggleMrkerButton_clicked)
toggleFoundPeaksButton.clicked.connect(toggleFoundPeaksButton_clicked)
toggleFoundCrystalButton.clicked.connect(toggleFoundCrystalButton_clicked)
zoomOnCrystalButton.clicked.connect(zoomOnCrystalButton_clicked)
reloadButton.clicked.connect(lambda: read_files())
#imageWidget.resize(800, 800)


topWidget = QtGui.QWidget()
layout = QtGui.QGridLayout()
topWidget.setLayout(layout)

layoutButtons = QtGui.QGridLayout()
layoutButtons.addWidget(nextImageButton, 0, 3)
layoutButtons.addWidget(previousImageButton, 0, 2)
layoutButtons.addWidget(plus10ImageButton, 0, 4)
layoutButtons.addWidget(minus10ImageButton, 0, 1)
layoutButtons.addWidget(randomImageButton, 0, 0)
layoutButtons.addWidget(lastImageButton, 0, 5)
layoutButtons.addWidget(reloadButton, 0, 10)
layoutButtons.addWidget(toggleMarkerButton, 0, 20)
layoutButtons.addWidget(toggleFoundPeaksButton, 0, 21)
layoutButtons.addWidget(toggleFoundCrystalButton, 0, 22)
layoutButtons.addWidget(zoomOnCrystalButton, 0, 23)

layout.addWidget(imageWidget, 0, 0)
layout.addWidget(mapWidget, 0, 1)
layout.addLayout(layoutButtons, 1, 0, 1, 2)

topWidget.show()

show_map = True
show_peaks = True
show_predict = True
show_markers = True
map_zoomed = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Viewer for Serial Electron Diffraction data')
    parser.add_argument('filename', type=str, help='Stream file, list file, or HDF5')
    parser.add_argument('-g', '--geometry', type=str, help='CrystFEL geometry file, might be helpful')
    parser.add_argument('-q', '--query', type=str, help='Query string to filter shots by column values')
    parser.add_argument('-d', '--data_path', type=str, help='Data field in HDF5 file(s). Defaults to stream file or tries a few.')
    parser.add_argument('--map_path', type=str, help='Path to map image', default='/%/map/image')
    parser.add_argument('--feature_path', type=str, help='Path to map feature table', default='/%/map/features')
    parser.add_argument('--peaks_path', type=str, help='Path to peaks table', default='/%/results/peaks')
    parser.add_argument('--predict_path', type=str, help='Path to prediction table', default='/%/results/predict')
    parser.add_argument('--no_map', help='Hide map, even if we had it', action='store_true')
    parser.add_argument('--beam_diam', type=int, help='Beam size displayed in real space, in pixels', default=5)

    args = parser.parse_args()

    # operation modes:
    # (1) file list (+ geometry) + nxs: estimate geometry from nxs if geometry is absent
    # (2) expanded file list (+ geometry) + nxs: first match nxs shot lists vs expanded file list
    # (3) (expanded) file list + geometry + hdf5: omit map image automatically
    # (4) stream + nxs: as (2), peaks/predict in stream take precedence over nxs
    # (5) stream + hdf5: as (3)

    # TODO next: work on read_file
    read_files()

    switch_shot(0)

    tmp = rawImage.copy().ravel()
    tmp.sort()
    level_min = tmp[round(0.02 * tmp.size)]
    level_max = tmp[round(0.98 * tmp.size)] * 2
    hist_img.setLevels(level_min, level_max)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
