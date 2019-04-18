import hdf5plugin
import h5py
import numpy as np
import time
from lambda_tools.io import *

import matplotlib.pyplot as plt
# for interactive pycharm: %matplotlib tk
import pyqtgraph as pg

import importlib
import sys

PyQt4_found = importlib.util.find_spec("PyQt4")
if PyQt4_found is not None:
    from PyQt4 import QtGui, QtCore
else:
    from PyQt5 import QtGui, QtCore

from lambda_tools.StreamFileParser import *

pg.setConfigOptions(imageAxisOrder='row-major')

# all the stuff that should go to an option parser:
data_path = '/%/data'
img_array_list = ['centered_fr', 'centered', 'raw_counts']
#img_array_list = ['centered', 'raw_counts']
map_path = '/%/map'
result_path = '/%/results'
beam_diam = 5
show_map = True
show_peaks = True
show_predict = True
show_markers = True
map_zoomed = True
query = 'frame == 1'

# operation modes:
# (1) only list and NeXus files
# (2) stream file plus NeXus
# (3) stream plus plain HDF5
# Yaroslav's version does (3), ours (1), for now.

def read_files(filename):
    global shots, features, peaks, predict

    shots = get_meta_lists(filename, data_path, 'shots')['shots']
    nshots0 = shots.shape[0]
    shots = shots.loc[shots['frame'] >= 0, :].query(query).reset_index()
    nshots = shots.shape[0]
    print(f'{nshots} shots from {nshots0} selected')

    try:
        features = get_meta_lists(filename, map_path, 'features')['features']
    except KeyError:
        print(f'No mapping features found in file {filename}')
        features = None
    try:
        peaks = get_meta_lists(filename, result_path, 'peaks')['peaks']
    except KeyError:
        print(f'No peaks found in file {filename}')
        peaks = None
    try:
        predict = get_meta_lists(filename, result_path, 'predict')['predict']
    except KeyError:
        print(f'No prediction spots found in file {filename}')
        predict = None

    if ('shot_in_subset' not in shots.columns) and 'Event' in shots.columns:
        shots[['shot_in_subset', 'subset']] = shots['Event'].str.split('//')


# CALLBACK FUNCTIONS

def switch_shot(serial):
    global imageSerialNumber, shot
    imageSerialNumber = max(0, serial % shots.shape[0])
    shot = shots.iloc[imageSerialNumber, :]
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
    global imageSerialNumber, rawImage, mapImage, shot

    try:
        with h5py.File(shot['file']) as f:
            for img_array in img_array_list:
                try:
                    path = data_path.replace('%', shot['subset']) + '/' + img_array
                    rawImage = f[path][int(shot['shot_in_subset']), ...]
                    print('Loading {}:{} from {}'.format(path, shot['shot_in_subset'], shot['file']))
                    break
                except KeyError:
                    continue
            else:
                raise KeyError('None of the stack names {} found'.format(img_array_list))

            if show_map:
                mapImage = f[map_path.replace('%', shot['subset']) + '/image'][...]
    except Exception as err:
        print('Could not load image data due to {}'.format(err))
        rawImage = rawImage

def updatePlot():
    global img, mapimg, hist, imageSerialNumber, rawImage, mapImage, shot, map_zoomed

    if show_peaks and (peaks is not None) and show_markers:
        ring_pen = pg.mkPen('g', width=2)
        found_peak_canvas.setData(peaks.loc[peaks['serial'] == shot.name, 'fs/px'] + 0.5,
                                  peaks.loc[peaks['serial'] == shot.name, 'ss/px'] + 0.5,
                                  symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), antialias=True)

    else:
        found_peak_canvas.clear()

    if show_predict and (predict is not None) and show_markers:
        ring_pen = pg.mkPen('g', width=2)
        predicted_peak_canvas.setData(predict[peaks['serial'] == shot, 'fs/px'] + 0.5,
                                      predict[peaks['serial'] == shot, 'ss/px'] + 0.5,
                                      symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), antialias=True)

    else:
        predicted_peak_canvas.clear()

    if features is not None and show_markers:
        ring_pen = pg.mkPen('g', width=2)
        dot_pen = pg.mkPen('y', width=0.5)

        region_feat = features.loc[(features['subset'] == shot['subset']) &
                                   (features['file'] == shot['file']), :]

        if shot['crystal_id'] != -1:
            single_feat = region_feat.loc[region_feat['crystal_id'] == shot['crystal_id'], :]
            found_features_canvas.setData(region_feat['crystal_x'], region_feat['crystal_y'],
                                          symbol='+', size=7, pen=dot_pen, brush=(0, 0, 0, 0), pxMode=True)
        #found_features_canvas.setData([shot['crystal_x'],], [shot['crystal_y'],],
        #                              symbol='+', size=13, pen=dot_pen, brush=(0, 0, 0, 0), pxMode=True)

            if map_zoomed:

                p2.setRange(xRange=(single_feat['crystal_x'].values - 5*beam_diam, single_feat['crystal_x'].values + 5*beam_diam),
                                   yRange=(single_feat['crystal_y'].values - 5*beam_diam, single_feat['crystal_y'].values + 5*beam_diam))
                single_feature_canvas.setData(single_feat['crystal_x'], single_feat['crystal_y'],
                                              symbol='o', size=beam_diam, pen=ring_pen, brush=(0, 0, 0, 0), pxMode=False)
            else:
                single_feature_canvas.setData(single_feat['crystal_x'], single_feat['crystal_y'],
                                              symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0), pxMode=True)
                p2.setRange(xRange=(0, mapImage.shape[1]), yRange=(0, mapImage.shape[0]))

        else:
            single_feature_canvas.setData([],[])

    levels = hist.getLevels()
    img.setImage(rawImage, autoRange=False)
    img.setLevels(levels)
    mapimg.setImage(mapImage)
    hist.setLevels(levels[0], levels[1])


def update():
    global imageSerialNumber

    found_peak_canvas.clear()
    predicted_peak_canvas.clear()
    pg.QtGui.QApplication.processEvents()

    updateImage()
    updatePlot()

    topWidget.setWindowTitle('{} Reg {} Run {} Feat {} Frame {} ({}//{} in {}, {} out of {}) '.format(shot['sample'],
                                                                                         shot['region'], shot['run'],
                                                                                         shot['crystal_id'],
                                                                                         shot['frame'],
                                                                                         shot['subset'],
                                                                                         shot['shot_in_subset'],
                                                                                         shot['file'],
                                                                                                      shot.name, shots.shape[0]))

########################################################## gui

pg.mkQApp()

imageWidget = pg.GraphicsLayoutWidget()
imageWidget.setWindowTitle('stream file viewer')

# A plot area (ViewBox + axes) for displaying the image
p1 = imageWidget.addViewBox()
p1.setAspectLocked()

img = pg.ImageItem()
img.setZValue(0)
p1.addItem(img)

found_peak_canvas = pg.ScatterPlotItem()
p1.addItem(found_peak_canvas)
found_peak_canvas.setZValue(2)

predicted_peak_canvas = pg.ScatterPlotItem()
p1.addItem(predicted_peak_canvas)
predicted_peak_canvas.setZValue(2)

# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
imageWidget.addItem(hist)

mapWidget = pg.GraphicsLayoutWidget()
mapWidget.setWindowTitle('region map')

# Map image control
p2 = mapWidget.addViewBox()
p2.setAspectLocked()

mapimg = pg.ImageItem()
mapimg.setZValue(0)
p2.addItem(mapimg)

found_features_canvas = pg.ScatterPlotItem()
p2.addItem(found_features_canvas)
found_features_canvas.setZValue(2)

single_feature_canvas = pg.ScatterPlotItem()
p2.addItem(single_feature_canvas)
single_feature_canvas.setZValue(2)

# Contrast/color control
hist2 = pg.HistogramLUTItem()
hist2.setImageItem(mapimg)
mapWidget.addItem(hist2)

# Control Buttons

topWidget = QtGui.QWidget()
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
randomImageButton.clicked.connect(lambda: switch_shot(np.random.randint(0, shots.shape[0]-1)))
plus10ImageButton.clicked.connect(lambda: switch_shot_rel(+10))
minus10ImageButton.clicked.connect(lambda: switch_shot_rel(-10))
lastImageButton.clicked.connect(lambda: switch_shot(shots.index.max()))
toggleMarkerButton.clicked.connect(toggleMrkerButton_clicked)
toggleFoundPeaksButton.clicked.connect(toggleFoundPeaksButton_clicked)
toggleFoundCrystalButton.clicked.connect(toggleFoundCrystalButton_clicked)
zoomOnCrystalButton.clicked.connect(zoomOnCrystalButton_clicked)
reloadButton.clicked.connect(lambda: read_files(filename))
#imageWidget.resize(800, 800)

layout = QtGui.QGridLayout()
layoutButtons = QtGui.QGridLayout()
topWidget.setLayout(layout)
layoutButtons.addWidget(nextImageButton, 0, 0)
layoutButtons.addWidget(previousImageButton, 0, 1)
layoutButtons.addWidget(plus10ImageButton, 0, 2)
layoutButtons.addWidget(minus10ImageButton, 0, 3)
layoutButtons.addWidget(randomImageButton, 0, 4)
layoutButtons.addWidget(lastImageButton, 0, 5)
layoutButtons.addWidget(reloadButton, 0, 10)
layoutButtons.addWidget(toggleMarkerButton, 0, 20)
layoutButtons.addWidget(toggleFoundPeaksButton, 0, 21)
layoutButtons.addWidget(toggleFoundCrystalButton, 0, 22)
layoutButtons.addWidget(zoomOnCrystalButton, 0, 23)
layout.addWidget(imageWidget, 0, 0)
layout.addLayout(layoutButtons, 1, 0, 1, 2)
layout.addWidget(mapWidget, 0, 1)

topWidget.show()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("need a list file or NeXus-compliant HDF5")
        exit

    filename = sys.argv[1]

    if len(sys.argv) > 2:
        query = sys.argv[2]

    if len(sys.argv) > 3:
        img_array_list = [sys.argv[3]]

    read_files(filename)

    switch_shot(0)
    update()
    tmp = rawImage.copy().ravel()
    tmp.sort()
    level_min = tmp[round(0.02 * tmp.size)]
    level_max = tmp[round(0.98 * tmp.size)] * 2
    hist.setLevels(level_min, level_max)
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
