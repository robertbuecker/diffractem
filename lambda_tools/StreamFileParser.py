from .cfelpyutils import crystfel_utils, geometry_utils
import os
from collections import namedtuple
import numpy as np

# Stream file parser from Yaroslav

streamFileChunk_t = namedtuple(
    typename='streamFileChunk',
    field_names=['filename', 'subset', 'Event', "Image_serial_number", 'foundPeaks', 'crystals']
)

foundPeaks_t = namedtuple(
    typename='foundPeaks',
    field_names=['fs', 'ss', 'radius', 'intensity']
)

crystal_t = namedtuple(
    typename='crystal',
    field_names=['fs', 'ss', 'basis_inv']  # fs,ss are lists
)

def getImageSerialNumber(streamFileChunk):
    return streamFileChunk.Image_serial_number

class StreamFileParser:
    def __init__(self, filename, maxChunksToParse=1e6, scratchdir='.'):
        self.streamFileChunks = []
        with open(filename, "r") as myfile:
            streamFileLines = myfile.readlines()

        geometryFileName = scratchdir + "/geometry.tmp___61321572672345677543624725625672"

        with open(geometryFileName, "w") as geometryFile:
            currentLine = 0
            while streamFileLines[currentLine] != "----- Begin geometry file -----\n":
                currentLine += 1
            currentLine += 1

            while streamFileLines[currentLine] != "----- End geometry file -----\n":
                geometryFile.write(streamFileLines[currentLine])
                currentLine += 1
            currentLine += 1

        self.geometry = crystfel_utils.load_crystfel_geometry(geometryFileName)
        os.remove(geometryFileName)

        while currentLine < len(streamFileLines) and len(self.streamFileChunks) < maxChunksToParse:
            crystals = []

            while currentLine < len(streamFileLines) and streamFileLines[currentLine] != "----- Begin chunk -----\n":
                currentLine += 1
            if currentLine == len(streamFileLines):
                break
            currentLine += 1
            filename = streamFileLines[currentLine].split()[2]
            currentLine += 1

            if "Event" in streamFileLines[currentLine]:
                Event = int(streamFileLines[currentLine].split()[1].split('//')[1])
                subset = streamFileLines[currentLine].split()[1].split('//')[0]
                currentLine += 1
            else:
                Event = None
                subset = None

            Image_serial_number = int(streamFileLines[currentLine].split()[3])
            currentLine += 1

            while "num_peaks" not in streamFileLines[currentLine]:
                currentLine += 1
            peakCount = int(streamFileLines[currentLine].split()[2])
            currentLine += 1

            while streamFileLines[currentLine] != "Peaks from peak search\n":
                currentLine += 1
            currentLine += 2

            foundPeaks = foundPeaks_t(np.zeros(peakCount), np.zeros(peakCount), np.zeros(peakCount), np.zeros(peakCount))
            for i in range(peakCount):
                tmp = np.fromstring(streamFileLines[currentLine], dtype=float, sep=' ')
                foundPeaks.fs[i] = tmp[0]
                foundPeaks.ss[i] = tmp[1]
                foundPeaks.radius[i] = tmp[2]
                foundPeaks.intensity[i] = tmp[3]
                currentLine += 1

            crystals = []
            while True:
                while streamFileLines[currentLine] != "--- Begin crystal\n" and streamFileLines[currentLine] != "----- End chunk -----\n":
                    currentLine += 1
                if streamFileLines[currentLine] == "----- End chunk -----\n":
                    break
                currentLine += 1

                basis = np.zeros((3,3))

                while "astar" not in streamFileLines[currentLine]:
                    currentLine += 1
                tmp = streamFileLines[currentLine].split()
                basis[0][0] = tmp[2]
                basis[1][0] = tmp[3]
                basis[2][0] = tmp[4]
                currentLine += 1
                tmp = streamFileLines[currentLine].split()
                basis[0][1] = tmp[2]
                basis[1][1] = tmp[3]
                basis[2][1] = tmp[4]
                currentLine += 1
                tmp = streamFileLines[currentLine].split()
                basis[0][2] = tmp[2]
                basis[1][2] = tmp[3]
                basis[2][2] = tmp[4]

                while "num_reflections" not in streamFileLines[currentLine]:
                    currentLine += 1
                peakCount = int(streamFileLines[currentLine].split()[2])

                while streamFileLines[currentLine] != "Reflections measured after indexing\n":
                    currentLine += 1
                currentLine += 2

                crystal = crystal_t(np.zeros(peakCount), np.zeros(peakCount), basis)
                for i in range(peakCount):
                    tmp = np.fromstring(streamFileLines[currentLine], dtype=float, sep=' ')
                    crystal.fs[i] = tmp[7]
                    crystal.ss[i] = tmp[8]
                    currentLine += 1

                crystals.append(crystal)

            self.streamFileChunks.append(streamFileChunk_t(filename, subset, Event, Image_serial_number, foundPeaks, crystals))

        self.streamFileChunks.sort(key=getImageSerialNumber)



    def getGeometry(self):
        return self.geometry
