import hdf5plugin
import h5py
import numpy as np
import time

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

# streamFileParser = StreamFileParser("/gpfs/cfel/cxi/scratch/user/gevorkov/test_indexing_aps_jungfrau_protK/idx_protKdoc_chip1dot6_12bunch_run1.stream", 10)

if len(sys.argv) < 2:
    print("need path to stream file as argument!")
    exit

maxChunksToLoad = 1e10
if len(sys.argv) >= 3:
    maxChunksToLoad = int(sys.argv[2])

streamFileParser = StreamFileParser(sys.argv[1], maxChunksToLoad)

geometry = streamFileParser.getGeometry()
pixel_maps = geometry_utils.compute_pixel_maps(geometry)

first_panel = list(geometry['panels'].keys())[0]
pixel_length = 1 / float(geometry['panels'][first_panel]['res'])
pixel_maps_m = namedtuple(  # pylint: disable=C0103
    typename='detectorPositions',
    field_names=['y', 'x']
)
pixel_maps_m.x = pixel_maps.x * pixel_length
pixel_maps_m.y = pixel_maps.y * pixel_length
pixel_maps_for_visualization = geometry_utils.adjust_pixel_maps_for_pyqtgraph(pixel_maps)
alignedImage = np.zeros(
    shape=geometry_utils.compute_minimum_array_size(pixel_maps),
    dtype=np.float32  # pylint: disable=E1101
)
visualization_offset = namedtuple(  # pylint: disable=C0103
    typename='offsets',
    field_names=['y', 'x']
)
visualization_offset.x = pixel_maps_for_visualization.x[0][0] - pixel_maps.x[0][0]
visualization_offset.y = pixel_maps_for_visualization.y[0][0] - pixel_maps.y[0][0]

dataPathInFile = geometry['panels'][first_panel]['data']
mapPathInFile = '/%/map/image'

if dataPathInFile is None:
    print("data location has to be written in geometry file!")
    exit

showMarker_flag = True
showFoundPeaks_flag = True
showFoundCrystal_flag = 2
crystalNumber = 0

imageSerialNumber = 1


def previousImageButton_clicked():
    global imageSerialNumber, crystalNumber

    imageSerialNumber -= 1
    imageSerialNumber = imageSerialNumber % (len(streamFileParser.streamFileChunks) + 1)
    if imageSerialNumber == 0:
        imageSerialNumber = -1
    crystalNumber = 0

    update()


def nextImageButton_clicked():
    global imageSerialNumber, crystalNumber

    imageSerialNumber += 1
    imageSerialNumber = imageSerialNumber % (len(streamFileParser.streamFileChunks) + 1)
    if imageSerialNumber == 0:
        imageSerialNumber = 1
    crystalNumber = 0

    update()

def randomImageButton_clicked():
    global imageSerialNumber, crystalNumber

    imageSerialNumber = np.random.randint(1, len(streamFileParser.streamFileChunks))
    crystalNumber = 0

    update()


def minus10ImageButton_clicked():
    global imageSerialNumber, crystalNumber

    imageSerialNumber -= 10
    imageSerialNumber = imageSerialNumber % (len(streamFileParser.streamFileChunks) + 1)
    if imageSerialNumber == 0:
        imageSerialNumber = -1
    crystalNumber = 0

    update()


def plus10Button_clicked():
    global imageSerialNumber, crystalNumber

    imageSerialNumber += 10
    imageSerialNumber = imageSerialNumber % (len(streamFileParser.streamFileChunks) + 1)
    if imageSerialNumber == 0:
        imageSerialNumber = 1
    crystalNumber = 0

    update()


def toggleMrkerButton_clicked():
    global showMarker_flag

    showMarker_flag = not showMarker_flag
    update()


def toggleFoundPeaksButton_clicked():
    global showFoundPeaks_flag

    showFoundPeaks_flag = not showFoundPeaks_flag
    update()


def toggleFoundCrystalButton_clicked():
    global showFoundCrystal_flag, crystalNumber

    showFoundCrystal_flag = (showFoundCrystal_flag + 1) % 3
    crystalNumber = 0
    update()


def previousCrystakButton_clicked():
    global crystalNumber

    crystalNumber -= 1
    crystalNumber = crystalNumber % len(streamFileParser.streamFileChunks[imageSerialNumber - 1].crystals)

    update()


def nextCrystalButton_clicked():
    global crystalNumber, imageSerialNumber

    crystalNumber += 1
    crystalNumber = crystalNumber % len(streamFileParser.streamFileChunks[imageSerialNumber - 1].crystals)

    update()


currentImageSerialNumber = 1e100


def updateImage():
    global imageSerialNumber, streamFileParser, dataPathInFile, rawImage, currentImageSerialNumber, mapImage

    if currentImageSerialNumber == imageSerialNumber:
        return
    currentImageSerialNumber = imageSerialNumber

    dataFilename = streamFileParser.streamFileChunks[imageSerialNumber - 1].filename
    event = streamFileParser.streamFileChunks[imageSerialNumber - 1].Event
    subset = streamFileParser.streamFileChunks[imageSerialNumber - 1].subset
    dataFile = h5py.File(dataFilename, "r")
    if event is not None:
        rawImage = np.array(dataFile[dataPathInFile.replace('%', subset)][event, ...], dtype=np.float32)
    else:
        rawImage = np.array(dataFile[dataPathInFile.replace('%', subset)][...], dtype=np.float32)

    if map is not None:
        mapImage = np.array(dataFile[mapPathInFile.replace('%', subset)][...], dtype=np.float32)

    print("image with serial number", imageSerialNumber, "loaded")
    print("file name", dataFilename, "subset", subset, "event", event)
    imageWidget.setWindowTitle(dataFilename)


def updatePlot():
    global img, mapimg, hist, runPeakFinder9_flag, alignedImage, pixel_maps_for_visualization, imageSerialNumber, streamFileParser, \
        rawImage, showMarker_flag, showFoundPeaks_flag, showFoundCrystal_flag, crystalNumber, mapImage

    if showFoundPeaks_flag and showMarker_flag:
        ring_pen = pg.mkPen('g', width=2)
        center_of_mass__raw_x = np.round(streamFileParser.streamFileChunks[imageSerialNumber - 1].foundPeaks.fs).astype(int)
        center_of_mass__raw_y = np.round(streamFileParser.streamFileChunks[imageSerialNumber - 1].foundPeaks.ss).astype(int)

        center_of_mass__raw_x = np.clip(center_of_mass__raw_x, 0, pixel_maps_for_visualization.x.shape[1] - 1)
        center_of_mass__raw_y = np.clip(center_of_mass__raw_y, 0, pixel_maps_for_visualization.y.shape[0] - 1)

        center_of_mass__aligned_x = pixel_maps_for_visualization.x[np.round(center_of_mass__raw_y).astype(int), np.round(center_of_mass__raw_x).astype(int)]
        center_of_mass__aligned_y = pixel_maps_for_visualization.y[np.round(center_of_mass__raw_y).astype(int), np.round(center_of_mass__raw_x).astype(int)]
        found_peak_canvas.setData(center_of_mass__aligned_y + 0.5, center_of_mass__aligned_x + 0.5, symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0))


        #debug!!!!!!
        #found_peak_canvas.addPoints(np.array([pixel_maps_for_visualization.x.shape[0]/2]), np.array([pixel_maps_for_visualization.x.shape[1]/2]), symbol='s', size=13, pen=ring_pen, brush=(0, 0, 0, 0))

    else:
        found_peak_canvas.clear()

    predicted_peak_canvas.clear()
    crystalsCount = len(streamFileParser.streamFileChunks[imageSerialNumber - 1].crystals)
    if crystalsCount > 0 and showFoundCrystal_flag > 0 and showMarker_flag:
        colors = ['r', 'c', 'm', 'y', 'g', 'b']

        if showFoundCrystal_flag == 1:
            crystalsRange = range(crystalNumber, crystalNumber + 1)
        else:
            crystalsRange = range(crystalsCount)

        for i in crystalsRange:
            ring_pen = pg.mkPen(colors[i % len(colors)], width=2)
            center_of_mass__raw_x = np.round(streamFileParser.streamFileChunks[imageSerialNumber - 1].crystals[i].fs).astype(int)
            center_of_mass__raw_y = np.round(streamFileParser.streamFileChunks[imageSerialNumber - 1].crystals[i].ss).astype(int)

            center_of_mass__raw_x = np.clip(center_of_mass__raw_x, 0, pixel_maps_for_visualization.x.shape[1] - 1)
            center_of_mass__raw_y = np.clip(center_of_mass__raw_y, 0, pixel_maps_for_visualization.y.shape[0] - 1)
            center_of_mass__aligned_x = pixel_maps_for_visualization.x[center_of_mass__raw_y, center_of_mass__raw_x]
            center_of_mass__aligned_y = pixel_maps_for_visualization.y[center_of_mass__raw_y, center_of_mass__raw_x]
            predicted_peak_canvas.addPoints(center_of_mass__aligned_y + 0.5, center_of_mass__aligned_x + 0.5, symbol='s', size=12, pen=ring_pen,
                                                brush=(0, 0, 0, 0))

            #debug!!!!!!
            #found_peak_canvas.addPoints(np.array([pixel_maps_for_visualization.x.shape[0] / 2 + 0.5]), np.array([pixel_maps_for_visualization.x.shape[1] / 2 + 0.5]),
            #                            symbol='o', size=13, pen=ring_pen, brush=(0, 0, 0, 0))

    levels = hist.getLevels()
    alignedImage = geometry_utils.apply_pixel_maps(rawImage, pixel_maps_for_visualization, output_array=alignedImage)
    img.setImage(alignedImage, autoRange=False)
    img.setLevels(levels)
    mapimg.setImage(mapImage)
    hist.setLevels(levels[0], levels[1])

    # debug!!!!!!
    #found_peak_canvas.addPoints(np.array([alignedImage.shape[0] / 2]), np.array([alignedImage.shape[1] / 2]), symbol='s',
    #                            size=13, pen=ring_pen, brush=(0, 0, 0, 0))


def update():
    global imageSerialNumber

    found_peak_canvas.clear()
    predicted_peak_canvas.clear()
    pg.QtGui.QApplication.processEvents()

    updateImage()
    updatePlot()


def toggleMrkerButton_clicked():
    global showMarker_flag

    showMarker_flag = not showMarker_flag
    updatePlot()


########################################################## gui


pg.mkQApp()
topWidget = QtGui.QWidget()
nextImageButton = QtGui.QPushButton('next')
previousImageButton = QtGui.QPushButton('previous')
randomImageButton = QtGui.QPushButton('random')
plus10ImageButton = QtGui.QPushButton('plus 10')
minus10ImageButton = QtGui.QPushButton('minus 10')
toggleMarkerButton = QtGui.QPushButton('toggle marker')
toggleFoundPeaksButton = QtGui.QPushButton('toggle found \npeaks markers')
toggleFoundCrystalButton = QtGui.QPushButton('toggle found \ncrystal markers mode')
nextCrystalButton = QtGui.QPushButton('next crystal')
previousCrystakButton = QtGui.QPushButton('previous crystal')

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

found_features = pg.ScatterPlotItem()
p2.addItem(found_features)
found_peak_canvas.setZValue(2)

# Contrast/color control
hist2 = pg.HistogramLUTItem()
hist2.setImageItem(mapimg)
mapWidget.addItem(hist2)

nextImageButton.clicked.connect(nextImageButton_clicked)
previousImageButton.clicked.connect(previousImageButton_clicked)
randomImageButton.clicked.connect(randomImageButton_clicked)
plus10ImageButton.clicked.connect(plus10Button_clicked)
minus10ImageButton.clicked.connect(minus10ImageButton_clicked)
toggleMarkerButton.clicked.connect(toggleMrkerButton_clicked)
toggleFoundPeaksButton.clicked.connect(toggleFoundPeaksButton_clicked)
toggleFoundCrystalButton.clicked.connect(toggleFoundCrystalButton_clicked)
previousCrystakButton.clicked.connect(previousCrystakButton_clicked)
nextCrystalButton.clicked.connect(nextCrystalButton_clicked)

#imageWidget.resize(800, 800)

layout = QtGui.QGridLayout()
layoutButtons = QtGui.QGridLayout()
topWidget.setLayout(layout)
layoutButtons.addWidget(nextImageButton, 0, 0)
layoutButtons.addWidget(previousImageButton, 0, 1)
layoutButtons.addWidget(plus10ImageButton, 0, 2)
layoutButtons.addWidget(minus10ImageButton, 0, 3)
layoutButtons.addWidget(randomImageButton, 0, 4)
layoutButtons.addWidget(toggleMarkerButton, 0, 5)
layoutButtons.addWidget(toggleFoundPeaksButton, 0, 6)
layoutButtons.addWidget(toggleFoundCrystalButton, 0, 7)
layoutButtons.addWidget(nextCrystalButton, 0, 8)
layoutButtons.addWidget(previousCrystakButton, 0, 9)
layout.addWidget(imageWidget, 0, 0)
layout.addLayout(layoutButtons, 1, 0, 1, 2)
layout.addWidget(mapWidget, 0, 1)

topWidget.show()

if __name__ == '__main__':
    update()
    tmp = rawImage.copy().ravel()
    tmp.sort()
    level_min = tmp[round(0.02 * tmp.size)]
    level_max = tmp[round(0.98 * tmp.size)] * 2
    hist.setLevels(level_min, level_max)
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
