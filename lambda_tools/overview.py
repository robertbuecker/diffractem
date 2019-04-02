import json
from scipy.cluster.vq import kmeans2
from scipy._lib._util import _asarray_validated
from sklearn.cluster import KMeans
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
from skimage.segmentation import random_walker
from skimage.morphology import binary_erosion, binary_dilation, disk, watershed, \
    remove_small_holes, remove_small_objects, binary_closing
from skimage.transform import matrix_transform
from tifffile import imread, imsave
import os
import h5py

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


def whiten(obs, check_finite=False):
    """
    Adapted from c:/python27/lib/site-packages/skimage/filters/thresholding.py
        to return array and std_dev
    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = np.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        raise RuntimeWarning("Some columns have standard deviation zero. The values of these columns will not change.")
    return obs / std_dev, std_dev


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
        self._subset = subset
        self.labels = np.ndarray((0,0))
        self.transform_matrix = []
        self.mask = np.ndarray((0,0))
        self.crystals = pd.DataFrame()
        self._shots = pd.DataFrame()
        self.reference = None
        self.coordinate_source = 'none'
        self._detector_file = None
        self.meta = None
        self.meta_diff = None
        self._img_label = None

        if basename is not None:
            self.read(basename)

    @property
    def subset(self):
        if self._subset is not None:
            return self._subset
        else:
            return 'overview_{:03d}_{:03d}'.format(self.region_id, self.run_id)

    @subset.setter
    def subset(self, value):
        self._subset = value

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
        self._shots = pd.DataFrame()
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

    @property
    def shots(self):
        cols = [c for c in ['crystal_id', 'crystal_x', 'crystal_y']
                if c in self.crystals.columns]

        retsh = self._shots.merge(self.crystals.loc[:,cols],
                                 on='crystal_id', how='left')
        retsh['run'] = self.run_id
        retsh['region'] = self.region_id
        retsh['shot'] = range(len(retsh))
        retsh['subset'] = self.subset
        retsh['stem'] = self._img_label
        if self.mask.size:
            retsh['mask'] = self._img_label
        if self.detector_file is not None:
            retsh['file'] = self.detector_file
        return retsh

    @shots.setter
    def shots(self, value):
        coords_in_shots = ('crystal_x' in value.columns) and ('crystal_y' in value.columns)
        if len(self.crystals) > 0:
            if not value['crystal_id'].isin(self.crystals['crystal_id'].append(pd.Series(-1))).all():
                raise ValueError('All crystal_id values in the shot list must be in the overview crystal list.')
            if coords_in_shots:
                print('Crystal coordinates stored in the shot list will be ignored!')

        elif coords_in_shots:
            print('Setting crystal coordinates from shot list!')
            self.crystals = value.query('crystal_id >= 0').drop_duplicates(subset='crystal_id').\
                                loc[:,['crystal_id', 'crystal_x', 'crystal_y']].reset_index(drop=True)

        self._shots = value.loc[:,['crystal_id', 'pos_x', 'pos_y', 'frame']]

    @property
    def detector_file(self):
        # this may be a place to mangle with directories
        return self._detector_file

    @detector_file.setter
    def detector_file(self, value):
        # this may be a place to mangle with directories...
        # strip extension
        if '.nxs' in value:
            value = value.rsplit('.',1)[0]
        self._detector_file = value
        #try:
        fn = value + '.json'
        if os.path.isfile(fn):
            print('Reading file {}'.format(fn))
            self._meta_diff = json.load(open(fn))

    @property
    def meta(self):
        ret = json_normalize(self._meta, sep='_')
        ret.columns = [
            cn.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_')
            for cn in ret]
        return ret

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def meta_diff(self):
        ret = json_normalize(self._meta_diff, sep='_')
        ret.columns = [
            cn.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_')
            for cn in ret]
        return ret

    @meta_diff.setter
    def meta_diff(self, value):
        self._meta_diff = value

    def get_regionprops(self):
        return regionprops(self.labels, self.img, cache=True, coordinates='rc')

    def find_particles(self, show_plot=True, show_segments=True,
                       thr_fun=threshold_li, thr_offset=0, local=False, disk_size=49, two_pass=False,  # thresholding
                       morph_method='legacy', morph_disk=2, remove_carbon_lacing=False,  # morphology
                       segmentation_method='distance-watershed', min_dist=8,  # segmentation
                       picking_method='region-centroid', beam_radius=5,  # picking
                       **kwargs):
        """
        Crystal finding algorithm. Attempts to find images in 4 steps:
        (1) thresholding, yielding a binarized image. Can be done using a global or local (adaptive) threshold.
            Usually, global works better for STEM images, unless the background is very inhomogeneous. If there are
            bright-ish "blobs" containing the particles, two_pass should be used.
        (2) morphological operations, eliminating small features, filling holes etc.. Different sequences of operations
            are provided which are worth trying out
        (3) segmentation of the bright regions found by the previous steps, using watershed or random walker schemes
        (4) generation of crystal coordinates, either using segment centroids, or by filling up the segments with a
            number of points derived from a typical spacing (beam radius)

        :param show_plot: show a plot in the end to assess the result
        :param thr_fun: function used for global thresholding. Anything that takes the image as only positional argument
            can be inserted here (e.g. those from skimage.filters.thresholding). Defaults to threshold_li
        :param thr_offset: additional offset for found global threshold. Note that the images are normalized to the
            full 16bit range initially, so typically this value is in the range of 1000s
        :param local: use local thresholding instead, using the threshold_local function in skimage.filters. Usually
            a global threshold with two_pass=True works better.
        :param disk_size: averaging disk range, when local=True
        :param two_pass: use a two-pass thresholding scheme, where after thresholding and morphological filtering, a
            separate local threshold is found for each bright region. Then the thresholding is repeated. Often this
            is far superior to using a local threshold.
        :param morph_method: selects a sequence of morphological operations. Current options are 'legacy' and
            'instamatic', the latter derived from Stef Smeets' instamatic package. Just try what works best.
        :param morph_disk: disk radius for morphological operations.
        :param remove_carbon_lacing: attempts to remove lacey carbon artifacts. Only works if morph_method is set
            to 'instamatic'
        :param segmentation_method: method for segmentation. Options are 'distance-watershed', 'intensity-watershed',
            and 'random-walker'. The latter is not well tested yet. Usually, 'distance-watershed' is better for clearly
            visible particles that aggregate slightly, and 'intensity-watershed' for particles within larger blobs.
        :param min_dist: minimum distance between initial points for segmentation. Ideally corresponds to minimum
            distance between crystals. Not used for random walker segmentation.
        :param picking_method: 'region-centroid' and 'intensity-centroid' will place a single coordinate marker for
            acquisition into each segment; the latter weighs the pixels by their intensity. 'k-means' will place
            many coordinate markers on the segment, approximately spaced by beam_radius/2; this is the 'brute-force'
            option.
        :param beam_radius: spacing between coordinates when using 'brute-force' acquisition
        :param kwargs:
        :return:
        """

        if 'intensity_centroid' in kwargs.keys():
            print('Please do not use intensity centroid option anymore, and picking_method instead!')
            if kwargs['intensity_centroid']:
                picking_method = 'intensity-centroid'
            else:
                picking_method = 'region-centroid'

        # STEP 1: thresholding
        def threshold(img):
            if local:
                # binarized = img > rank.otsu(img_as_ubyte(img), disk(disk_size))
                binarized = img > threshold_local(img, disk_size, method='mean')
            else:
                binarized = img > thr_fun(img) + thr_offset

            return binarized

        # STEP 2: morphology
        def morphology(binarized):
            if morph_method == 'legacy':
                morph = binary_dilation(binarized, disk(1))
                morph = binary_erosion(morph, disk(morph_disk))

            elif morph_method == 'instamatic':
                morph = remove_small_objects(binarized, min_size=4 * 4, connectivity=0)  # remove noise
                morph = binary_closing(morph, disk(morph_disk))  # dilation + erosion
                morph = binary_erosion(morph, disk(morph_disk))  # erosion
                if remove_carbon_lacing:
                    morph = remove_small_objects(morph, min_size=8 * 8, connectivity=0)
                    morph = remove_small_holes(morph, min_size=32 * 32, connectivity=0)
                morph = binary_dilation(morph, disk(morph_disk))  # dilation

            elif (morph_method is None) or (morph_method=='none'):
                morph = binarized

            else:
                raise ValueError('Unknown morphology method {}'.format(morph_method))

            return morph

        img = rescale_intensity(self.img, in_range='image')
        binarized = threshold(img)
        morph = morphology(binarized)

        if two_pass:
            lmorph = label(morph)
            #thr_loc = np.ones_like(img)*(thr_fun(img) + thr_offset)
            thr_loc = np.ones_like(img) * img.max()
            for ii in range(1, lmorph.max()):
                mask = lmorph == ii
                if mask.sum() < 3:
                    continue
                thr_loc[mask] = thr_fun(img[mask])

            binarized = img > thr_loc
            morph = morphology(binarized)

        # get background pixels
        bkg = np.invert(binary_dilation(morph, disk(morph_disk * 2)) | morph)

        # STEP 3: segmentation
        if segmentation_method == 'distance-watershed':
            distance = ndi.distance_transform_edt(morph)
            local_max = peak_local_max(distance, indices=False,
                                       min_distance=min_dist, labels=label(morph), exclude_border=True)
            label_image = label(local_max)
            self.labels = watershed(-distance, label_image, mask=morph)

        elif segmentation_method == 'intensity-watershed':
            distance = ndi.distance_transform_edt(morph)
            local_max = peak_local_max(img, indices=False, min_distance=min_dist, labels=label(morph))
            label_image = label(local_max)
            self.labels = watershed(img.max()-img, label_image, mask=morph)

        elif segmentation_method == 'random-walker':
            # From instamatic. Works a bit different from watershed version currently, as it adds another "background"
            # label. Still room for improvement...
            markers = morph * 2 + bkg
            self.labels = label(random_walker(img, markers, beta=50, spacing=(5, 5), mode='bf')) - 1
            #self.labels = segmented.astype(int) - 1

        else:
            raise ValueError('Unknown segmentation method {}'.format(segmentation_method))

        props = self.get_regionprops()

        # STEP 4: get crystal coordinates from segments
        if picking_method == 'intensity-centroid':
            self.coordinates = np.array([p.weighted_centroid for p in props])
            self.crystals['segment_id'] = self.crystals['crystal_id']

        elif picking_method == 'region-centroid':
            self.coordinates = np.array([p.centroid for p in props])
            self.crystals['segment_id'] = self.crystals['crystal_id']

        elif picking_method == 'k-means':
            coords = []
            segs = []
            for ii, prop in enumerate(props):
                area = prop.area
                bbox = np.array(prop.bbox)
                origin = bbox[0:2]
                ncoords = int(area // (np.pi*beam_radius**2)) + 1

                if ncoords > 1:
                    coordinates = np.argwhere(prop.image)
                    # kmeans needs normalized data (w), store std to calculate coordinates after
                    w, std = whiten(coordinates)
                    #km = KMeans(ncoords, max_iter=20, n_jobs=-1).fit(w)
                    #cluster_centroids = km.cluster_centers_
                    cluster_centroids, closest_centroids = kmeans2(w, ncoords, iter=20, minit='points')
                    coords.extend(cluster_centroids * std + origin[0:2])
                    segs.extend([ii]*len(cluster_centroids))
                else:
                    coords.append(prop.centroid)
                    segs.append(ii)

            self.coordinates = np.array(coords)
            self.crystals['segment_id'] = segs

        else:
            raise ValueError('Unknown coordinate method {}'.format(picking_method))

        self.mask = np.ndarray((0, 0)) # invalidate mask
        self.coordinate_source = 'picked'
        self.reference = None

        pdf = pd.DataFrame([{p: rp[p] for p in ['area', 'equivalent_diameter', 'major_axis_length',
                 'minor_axis_length', 'orientation']} for rp in props])

        self.crystals = self.crystals.merge(pdf, left_on='segment_id', right_index=True, how='left')

        print('{} particles found in {} segments.'.format(self.coordinates.shape[0], self.labels.max()))

        if show_plot:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))
            ax[0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
            ax[0].contour(binarized, [0.5], linewidths=0.5, colors="yellow")
            ax[0].contour(morph, [0.5], linewidths=0.5, colors="green")
            ax[0].set_title('ADF image')
            ax[1].imshow(img, cmap='gray', vmin=np.percentile(img, 1))
            if show_segments:
                ax[1].contour(self.labels, np.arange(self.labels.max()) + 0.5, linewidths=0.5, cmap='flag_r')

            ax[1].scatter(self.coordinates[:, 1], self.coordinates[:, 0], marker='o', facecolors='none', edgecolors='g',
                          s=20)
            #ax[1].contour(self.labels, np.arange(self.labels.max())+0.5, linewidths=0.5, colors='green')
            #for ii in range(self.labels.max()):
            #    ax[1].contour(self.labels == ii, [0.5], linewidths=0.5, colors='green')
            ax[1].set_title('{} coordinates, {} segments'.format(self.coordinates.shape[0], self.labels.max()))
            #plt.show()

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
            sh = tools.quantize_y_scan(sh, maxdev=y_pos_tol, min_rows=int(min(self.img.shape[0]/100, len(sh)-10)),
                                       max_rows=self.img.shape[0], inc=25)
        sh = tools.set_frames(sh, frames)
        sh = tools.insert_init(sh, predist=predist, dxmax=dxmax)
        self.shots = sh

    def export_scan_list(self, filename, delim=' '):
        scanpos = self.shots.loc[:,['pos_x', 'pos_y']] / self.img.shape[::-1]
        scanpos.to_csv(filename, sep=delim, header=False, index=False, line_terminator='\r\n')

    def make_mask(self, offset_x=0, offset_y=0, spotsize=0, pattern=None, init_cols=1, binary_fn=None):

        offset = np.array([offset_y, offset_x])
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

        if self._meta:
            fn = basename + '.json'
            print('Saving file {}'.format(fn))
            json.dump(self._meta, open(fn, 'w'), indent=4)

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
            self._meta = json.load(open(fn))
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

    @classmethod
    def from_h5(cls, filename, region_id=None, run_id=None, subset=None):
        """
        Load overview image data from HDF5 file. Caveat: acquisition metadata will be flattened!
        :param filename: obvious, right?
        :param region_id: if None (by default), assumes that the selected subset contains only one region/run
        :param run_id: if None (by default), assumes that the selected subset contains only one region/run
        :param subset: if None (by default), assumes that the file contains only a single subset
        :return: an OverviewImg object
        """

        meta = io.get_meta_lists(filename, flat=False)

        if subset is None:
            if len(meta) > 1:
                raise ValueError('File contains more than one subset. You have to specify which one to read.')
            else:
                subset = list(meta.keys())[0]

        meta = meta[subset]

        if (region_id is None) or (run_id is None):
            regrun = []
            for k, v in meta.items():
                if ('run' in v.columns) and ('region' in v.columns):
                    regrun.append(v[['region','run']])

            if not regrun:
                raise ValueError('Meta lists do not seem to contain region/run information. You have to give them explicitly!')

            regrun = pd.concat(regrun).drop_duplicates()
            if len(regrun) > 1:
                raise ValueError('Subset contains more than one region/run. You have to give it explicitly!')

            region_id = regrun.iloc[0, :]['region']
            run_id = regrun.iloc[0, :]['run']

        self = cls(region_id=region_id, run_id=run_id, subset=subset)

        if 'crystals' in meta.keys():
            cryst = meta['crystals']
            self.crystals = cryst.loc[(cryst['region']==self.region_id) & (cryst['run']==self.run_id), :]

        if 'shots' in meta.keys():
            self.shots = meta['shots'].loc[
                         (meta['shots']['region'] == self.region_id) & (meta['shots']['run'] == self.run_id), :]

        if len(self.crystals) == 0:
            raise ValueError('No crystals or shots list found in HDF5 file.')

        if 'stem_acqdata' in meta.keys():
            self.meta = meta['stem_acqdata'].loc[(meta['stem_acqdata']['region']==self.region_id) & (meta['stem_acqdata']['run']==self.run_id)]
            self.tilt = np.round(self.meta['Stage_A'].values * 180 / np.pi)

        if 'stem' in self.crystals.columns:
            self._img = io.get_meta_array(filename, 'stem', self.crystals.iloc[0, :], subset=self.subset)

        if 'mask' in self.crystals.columns:
            self.mask = io.get_meta_array(filename, 'mask', self.crystals.iloc[0, :], subset=self.subset)

        self.crystals.drop(['region', 'run'], axis=1, inplace=True)

        return self

    def to_h5(self, filename):
        """
        Write overview image data into a HDF5 file conforming to the downstream analysis codes.
        Note that this is a total mess so far.
        :param filename: output filename
        :return:
        """

        # all the implicit fields now have to be made explicit
        cryst = self.crystals.copy()
        cryst['region'] = self.region_id
        cryst['run'] = self.run_id
        lists = {'crystals': cryst}

        if self._meta is not None:
            m = self.meta.copy()
            m['region'] = self.region_id
            m['run'] = self.run_id
            lists.update({'stem_acqdata': m})

        if self._meta_diff is not None:
            md = self.meta_diff.copy()
            md['region'] = self.region_id
            md['run'] = self.run_id
            lists.update({'acqdata': md})

        # note: this has side-effects (but that's ok -> adds the STEM data set name)!
        self._img_label, _ = io.store_meta_array(filename, 'stem', {'region': self.region_id, 'run': self.run_id}, self.img, shots=cryst,
                            listname='crystals', subset_label=self.subset)

        if self.mask.size:
            io.store_meta_array(filename, 'mask', {'region': self.region_id, 'run': self.run_id}, self.mask, shots=cryst,
                                subset_label=self.subset)

        # must come last, so that all other parameters are properly set (shots is auto-generated)
        if self.shots.size:
            lists.update({'shots': self.shots})

        io.store_meta_lists(filename, {self.subset: lists}, flat=False) # MUST go last, as store_meta_array has side-effects