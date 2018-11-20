import json

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.io.json import json_normalize
from scipy import ndimage as ndi
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max, register_translation
from skimage.filters import *
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation, disk, watershed
from skimage.transform import matrix_transform
from tifffile import imread, imsave

from lambda_tools import tools, io


def align_ecc(img, img_ref, method='ecc', mode='affine',
              coords=None, rescale=False, use_gradient=True):

    if rescale:
        img0 = rescale_intensity(img_ref, in_range='image', out_range='float32').astype('float32')
        img1 = rescale_intensity(img, in_range='image', out_range='float32').astype('float32')
    else:
        img0 = img_ref.astype('float32')
        img1 = img.astype('float32')

    if use_gradient:

        def get_gradient(im):
            # Calculate the x and y gradients using Sobel operator
            grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
            grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

            # Combine the two gradients
            grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
            return grad

        img0 = get_gradient(img0)
        img1 = get_gradient(img1)

    shift = register_translation(img0, img1, 10)
    print('Found init shift: {}'.format(shift[0]))

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[:, 2] = -shift[0][::-1]
    # warp_matrix[:,2] = -shift[0]
    number_of_iterations = 1000000
    termination_eps = 1e-6
    ecc_mode = {'affine': cv2.MOTION_AFFINE, 'translation': cv2.MOTION_TRANSLATION}
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    # print(warp_matrix)
    (cc, warp_matrix) = cv2.findTransformECC(img0, img1, warp_matrix, ecc_mode[mode], criteria)
    # print(warp_matrix)
    img_x = cv2.warpAffine(img1, cv2.invertAffineTransform(warp_matrix), img1.shape)
    imgref_x = cv2.warpAffine(img0, warp_matrix, img0.shape)

    # make a scikit-image/ndimage-compatible output transform matrix (y and x flipped!)
    trans_matrix = np.vstack((np.hstack((np.rot90(warp_matrix[:2, :2], 2), np.flipud(warp_matrix[:, 2:]))), [0, 0, 1]))
    if coords is not None:
        coords_x = matrix_transform(coords, trans_matrix)
    else:
        coords_x = []

    return trans_matrix, coords_x, img_x, imgref_x


class OverviewImg:

    def __init__(self, img=None, coordinates=None, basename=None, region_id=0, run_id=0, subset=None):
        """
        Returns an OverviewImg object, which can contain a region overview image, along with particle coordinates,
        meta data, region properties, and a scan mask. It contains methods to extract coordinates by finding
        objects or transferring them from a reference image by an affine transform. Also, mask creation is included.
        The object can be read from or stored to a set of files with common prefixes.
        :param img: numpy array or tif file name. Can be used to set the image data.
        :param coordinates: numpy array or text file name. Can be used to set coordinate data.
        :param basename: string identifying the prefix of a set of files that (together) store the data of an overview
            image (as written by the store method)
        """


        if img is None:
            self._img = np.ndarray((0,0))
        elif isinstance(img, str):
            self._img = imread(img)
        elif isinstance(img, np.ndarray):
            self._img = img
        else:
            raise TypeError('Img must be a filename or a Numpy array')

        if coordinates is None:
            pass
        elif isinstance(coordinates, str):
            self.coordinates = np.loadtxt(coordinates)
        elif isinstance(coordinates, np.ndarray):
            self.coordinates = coordinates
        else:
            raise TypeError('Coordinates must be a filename or a Numpy array')

        self.defoc = 0
        self.tilt = 0
        self.region_id = region_id
        self.run_id = run_id
        if subset is None:
            self.subset = self.build_subset_name()
        else:
            self.subset = subset
        self.meta = {}
        self.labels = np.ndarray((0,0))
        self.transform_matrix = []
        self.mask = np.ndarray((0,0))
        self.crystals = pd.DataFrame()
        self.shots = pd.DataFrame()
        self.reference = None
        self.coordinate_source = 'none'

        if basename is not None:
            self.read(basename)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value
        # invalidate all learned data
        self.labels = np.ndarray((0,0))
        self.transform_matrix = []
        self.mask = np.ndarray((0,0))
        self.crystals = pd.DataFrame()
        self.shots = pd.DataFrame()
        self.coordinate_source = 'none'

    @property
    def coordinates(self):
        return self.crystals.loc[:, ['crystal_y', 'crystal_x', 'crystal_id']].values

    @coordinates.setter
    def coordinates(self, value):
        if value.shape[1] < 3:
            idx = np.arange(value.shape[0])
        else:
            idx = value[:,2]

        cryst = pd.DataFrame(value[:,:2], columns=['crystal_y', 'crystal_x'], index=idx)
        cryst['crystal_id'] = idx
        self.crystals = cryst

    def get_regionprops(self):
        return regionprops(self.labels, self.img, cache=True, coordinates='rc')

    def find_particles(self, morph_disk=1, show_plot=True, min_dist=8,
                       thr_fun=threshold_li, thr_offset=0, local=False, disk_size=49,
                       intensity_centroid=False):
        """

        :param morph_disk:
        :param show_plot:
        :param min_dist:
        :param thr_fun:
        :param local:
        :param disk_size:
        :param intensity_centroid:
        :return:
        """

        # TODO: THIS SO NEEDS IMPROVEMENT. Maybe steal from instamatic
        # ...especially the labeling (is that even needed?)
        # and segmentation steps are not yet helpful

        adf = rescale_intensity(self.img, in_range='image')
        thr_glob = thr_fun(adf) + thr_offset

        if local:
            # binarized = adf > rank.otsu(img_as_ubyte(adf), disk(disk_size))
            binarized = threshold_local(adf, disk_size, method='mean')
        else:
            binarized = adf > thr_glob

        adf2 = erosion(dilation(binarized, disk(1)), disk(morph_disk))
        # adf3 = dilation(adf2, disk(2))
        distance = ndi.distance_transform_edt(adf2)
        local_max = peak_local_max(distance, indices=False, min_distance=min_dist, labels=adf2)
        label_image = label(local_max)
        self.labels = watershed(-distance, label_image, mask=adf2)

        props = self.get_regionprops()

        if intensity_centroid:
            self.coordinates = np.array([p.weighted_centroid for p in props])
        else:
            self.coordinates = np.array([p.centroid for p in props])

        self.mask = np.ndarray((0, 0)) # invalidate mask
        self.coordinate_source = 'picked'
        self.reference = None

        pdf = pd.DataFrame([{p: rp[p] for p in ['area', 'equivalent_diameter', 'major_axis_length',
                 'minor_axis_length', 'orientation']} for rp in props],
                           index=self.crystals.index)

        self.crystals = pd.concat([self.crystals, pdf], axis=1)

        print('{} particles found.'.format(self.coordinates.shape[0]))

        if show_plot:
            fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 12))
            ax[0, 0].imshow(adf, cmap='inferno', vmin=np.percentile(adf, 1), vmax=np.percentile(adf, 99))
            ax[0, 0].set_title('ADF image')
            ax[0, 1].imshow(binarized, cmap='inferno')
            ax[0, 1].set_title('Threshold')
            ax[1, 0].imshow(adf2, cmap='inferno')
            ax[1, 0].set_title('Erosion')
            ax[1, 1].imshow(label_image, cmap='flag_r')
            ax[1, 1].set_title('Labeling')
            ax[2, 0].imshow(self.labels, cmap='flag_r')
            ax[2, 0].set_title('Segments')
            ax[2, 1].imshow(adf, cmap='gray', vmin=np.percentile(adf, 1), vmax=np.percentile(adf, 99))
            ax[2, 1].scatter(self.coordinates[:, 1], self.coordinates[:, 0], marker='o', facecolors='none', edgecolors='y', s=50)
            ax[2, 1].set_title('Coordinates')

    def align_overview(self, reference, show_plot=False,
                       method='ecc', use_gradient=False, transfer_props=False):
        #assert isinstance(reference, type(self))
        if self.tilt != 0:
            mode = 'affine'
        else:
            mode = 'translation'

        if method == 'ecc':
            self.transform_matrix, coords = \
                align_ecc(self.img, reference.img,
                          mode=mode, method='ecc', use_gradient=use_gradient,
                          coords=reference.coordinates[:,:2])[0:2]
        else:
            raise ValueError('Only ecc is allowed as method (for now)')

        self.crystals = reference.crystals.copy()
        self.crystals.loc[:,['crystal_y', 'crystal_x']] = coords

        # invalidate mask and labels
        self.mask = np.ndarray((0, 0))
        self.labels = np.ndarray((0, 0))
        self.coordinate_source = 'reference'
        self.reference = reference

        print('Transform is {}'.format(self.transform_matrix))

        if show_plot:
            _, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 3))
            reference.img_scatter_plot(ax.ravel()[0])
            self.img_scatter_plot(ax.ravel()[1])

    def make_scan_list(self, offset_x=0, offset_y=0, frames=1, y_pos_tol=1, predist=100, dxmax=300):
        # initialize by copying crystal list

        sh = self.crystals.loc[:,['crystal_id', 'crystal_x', 'crystal_y']].copy()
        sh['pos_x'] = sh['crystal_x'] + offset_x
        sh['pos_y'] = sh['crystal_y'] + offset_y
        inrange = (0 <= sh['pos_x']) & (sh['pos_x'] < self.img.shape[1]) & (0 <= sh['pos_y']) & (sh['pos_y'] < self.img.shape[0])
        sh = sh.loc[inrange,:]
        if y_pos_tol is not None:
            sh = tools.quantize_y_scan(sh, maxdev=y_pos_tol, min_rows=int(self.img.shape[0]/20), max_rows=self.img.shape[0])
        sh = tools.set_frames(sh, frames)
        sh = tools.insert_init(sh, predist=predist, dxmax=dxmax)
        self.shots = sh

    def export_scan_list(self, filename, delim=','):
        scanpos = self.shots.loc[:,['pos_x', 'pos_y']] / self.img.shape[::-1]
        scanpos.to_csv(filename, sep=delim, header=False, index=False)

    def make_mask(self, offset=[0,0], spotsize=0, pattern=None, init_cols=1, binary_fn=None):

        mask = np.zeros(self.img.shape).astype(np.int16)
        idcs = (self.coordinates[:,:2] + offset).round().astype(int)
        valid = (idcs[:, 0] >= 0) & (idcs[:, 1] >= 0) & (idcs[:, 0] < mask.shape[0]) & (idcs[:, 1] < mask.shape[1])
        mask[idcs[valid, 0], idcs[valid, 1]] = self.coordinates[valid, 2]

        if pattern is not None:
            mask = dilation(mask, pattern)

        mask = dilation(mask, disk(spotsize))
        mask[np.any(mask, axis=1), 0:init_cols] = -1

        if binary_fn:
            imsave(binary_fn, ((mask != 0) * 255).astype(np.int8))

        self.mask = mask

    def store(self, basename):

        if self.img.size:
            fn = basename + '.tif'
            print('Saving file {}'.format(fn))
            imsave(fn, self.img)

        if self.mask.size:
            fn = basename + '_mask.tif'
            print('Saving file {}'.format(fn))
            imsave(fn, self.mask)

        if self.meta:
            fn = basename + '.json'
            print('Saving file {}'.format(fn))
            json.dump(self.meta, open(fn, 'w'), indent=4)

        if self.coordinates.size:
            fn = basename + '_coord.txt'
            print('Saving file {}'.format(fn))
            np.savetxt(fn, self.coordinates, '%.8e')

    def read(self, basename):

        try:
            fn = basename + '.tif'
            print('Reading file {}'.format(fn))
            self.img = imread(fn)
        except:
            print('Could not read image file')

        try:
            fn = basename + '_mask.tif'
            print('Reading file {}'.format(fn))
            self.mask = imread(fn)
        except:
            print('Could not read mask file')

        try:
            fn = basename + '.json'
            print('Reading file {}'.format(fn))
            self.meta = json.load(open(fn))
        except:
            print('Could not read meta file')

        try:
            fn = basename + '_coord.txt'
            print('Reading file {}'.format(fn))
            self.coordinates = np.loadtxt(fn)
            self.coordinate_source = 'unknown'
            self.reference = None
        except:
            print('Could not read coordinate file')


    def img_scatter_plot(self, ax, colors=None):
        """
        Tool function to make a nice combined scatter and stem figure plot.
        :param ax: matplotlib axes object to plot into. Mandatory.
        :param colors: list of color values to be used for the scatter plot. Can be used to get consistent dot colors
            between many images (e.g. to check if tracking works)
        :return: colors (colors used for the scatter points), imgh (handle to image plot), scath (handle to scatter plot)
        """

        if colors is None:
            colors = self.coordinates[:,2]

        imgh = ax.imshow(self.img, cmap='gray', vmin=np.percentile(self.img, 1), vmax=np.percentile(self.img, 99))

        if len(self.coordinates) > 0:
            scath = ax.scatter(self.coordinates[:, 1], self.coordinates[:, 0], c=colors,
                               marker='o', facecolors='none', edgecolors='none',
                               s=50, alpha=0.3, cmap='Set1')
        else:
            scath = None

        return colors, imgh, scath

    def build_subset_name(self):
        return 'overview_{}_{}'.format(self.region_id, self.run_id)

    @classmethod
    def from_h5(cls, filename, region_id=0, run_id=0, subset=None):
        """
        Load overview image data from HDF5 file. Caveat: acquisition metadata will be flattened!
        :param filename:
        :param region_id:
        :param run_id:
        :param subset:
        :return:
        """
        self = cls(region_id=region_id, run_id=run_id, subset=subset)

        meta = io.get_meta_lists(filename, flat=False)[self.subset]

        if 'crystals' in meta.keys():
            cryst = meta['crystals']
            self.crystals = cryst.loc[(cryst['region']==self.region_id) & (cryst['run']==self.run_id),:]

        elif 'shots' in meta.keys():
            self.shots = meta['shots'].loc[(meta['shots']['region'] == self.region_id) & (meta['shots']['run'] == self.run_id),:]
            cryst = self.shots
            self.crystals = cryst.loc[(cryst['region'] == self.region_id) & (cryst['run'] == self.run_id) & (cryst['crystal_id'] >= 0),
                              ['crystal_y', 'crystal_x', 'crystal_id', 'region', 'run']].drop_duplicates()

        else:
            raise ValueError('No crystals or shots list found in HDF5 file.')

        if 'stem_acqdata' in meta.keys():
            self.meta = meta['stem_acqdata'].loc[(meta['stem_acqdata']['region']==self.region_id) & (meta['stem_acqdata']['run']==self.run_id)]
            self.tilt = np.round(self.meta['Stage_A'].values * 180 / np.pi)

        if 'stem' in cryst.columns:
            self.img = io.get_meta_array(filename, 'stem', cryst.iloc[0, :], subset=self.subset)

        if 'mask' in cryst.columns:
            self.mask = io.get_meta_array(filename, 'mask', cryst.iloc[0, :], subset=self.subset)

        self.crystals.drop(['region', 'run'], axis=1, inplace=True)

        return self

    def to_h5(self, filename):
        """
        Write overview image data into a HDF5 file conforming to the downstream analysis codes
        :param filename: output filename
        :param region_id: region id to set in the shot data
        :param run_id: run id to set in the shot data
        :param subset: subset label to set in the shot data
        :return:
        """

        if self.subset is None:
            subset = self.build_subset_name()
        else:
            subset = self.subset

        cryst = self.crystals.copy()
        cryst['region'] = self.region_id
        cryst['run'] = self.run_id
        self.meta.update({'region': self.region_id, 'run': self.run_id})
        acqdata = json_normalize(self.meta, sep='_')
        acqdata.columns = [cn.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_')
                           for cn in acqdata.columns]
        lists = {'crystals': cryst, 'stem_acqdata': acqdata}

        if self.shots.size:
            sh = self.shots.copy()
            sh['region'] = self.region_id
            sh['run'] = self.run_id
            lists.update({'shots': sh})

        io.store_meta_array(filename, 'stem', {'region': self.region_id, 'run': self.run_id}, self.img, shots=cryst,
                            subset_label=subset)
        if self.mask.size:
            io.store_meta_array(filename, 'mask', {'region': self.region_id, 'run': self.run_id}, self.mask, shots=cryst,
                                subset_label=subset)

        io.store_meta_lists(filename, {subset: lists}, flat=False) # MUST go last, as store_meta_array has side-effects