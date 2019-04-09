import json
import re
from glob import glob
from io import StringIO
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import tifffile
from dask import array as da

from pandas.io.json import json_normalize
from tifffile import imread

from lambda_tools.io import store_meta_array, store_meta_lists


def parse_acquisition_meta(shot_list, filename=None, include_meta=True, include_stem=True, include_mask=True,
                           subset_label=None, stem_postfix='_foc.tif', mask_postfix='_coll_mask.tif'):
    """
    Grabs the acquisition data from the JSON file, as well as optionally STEM and mask files.
    :param shot_list:
    :param filename:
    :param include_meta:
    :param include_stem:
    :param include_mask:
    :param subset_label:
    :param stem_postfix:
    :param mask_postfix:
    :return:
    """


    # iterate over file names, as they are unique, even between the subsets
    mdat = []
    for _, fn in shot_list.drop_duplicates(subset='file').iterrows():

        if 'subset' in fn.keys():
            subset_label = fn['subset']

        if include_meta:
            # read acquisition meta data JSON file
            newdat = json.load(open(fn['file'] + '.json'))
            newdat.update({'region': fn['region'], 'run': fn['run'], 'file': fn['file'], 'subset': subset_label})
            mdat.append(newdat)

        if filename and include_stem:
            store_meta_array(filename, 'stem', {'region': fn['region'], 'run': fn['run']},
                             imread(fn['file'] + stem_postfix), shot_list, subset_label=subset_label)

        if filename and include_mask:
            store_meta_array(filename, 'mask', {'region': fn['region'], 'run': fn['run']},
                             imread(fn['file'] + mask_postfix), shot_list, subset_label=subset_label)

    if include_meta:
        acqdata = json_normalize(mdat, sep='_')
        acqdata.columns = [cn.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('-', '_')
                           for cn in acqdata.columns]
        acqdata.set_index('file', drop=True, inplace=True)
    else:
        acqdata = None

    if filename is not None:
        if 'subset' in shot_list.columns:
            store_meta_lists(filename, {'shots': shot_list, 'acquisition_data': acqdata}, flat=True)
        else:
            store_meta_lists(filename, {subset_label: {'shots': shot_list, 'acquisition_data': acqdata}}, flat=False)

    return acqdata


def make_master_h5_legacy(files, master_name='master.h5', pxmask=None,
                   wavelength=0.025, distance=1, beam_center=(778, 258),
                   data_path='/entry/instrument/detector/data',
                   include_meta=True, meta_from=None,
                   new_scheme=False, stack=None):
    """
    Creates an Eiger-compatible master HDF5 file, which can be used to have eiger2cbf convert a set of Lambda .nxs
    files to CBF, in a form that can be digested by nXDS. All Metadata will be taken from the _first_ file!
    TODO: add metafile capabilities as needed
    :param files: list of files, or file pattern
    :param master_name: name of output file
    :param pxmask: dead pixel mask. Either as Numpy array, or as a tif filename.
    :return:
    """
    if isinstance(files, str):
        imgfiles = glob(files)
        imgfiles.sort()
    elif isinstance(files, list) or isinstance(files, tuple):
        imgfiles = files
    else:
        raise ValueError('files must be a file pattern or a list of files!')

    if pxmask is None:
        pass
    elif isinstance(pxmask, str):
        with tifffile.TiffFile(pxmask) as tif:
            pxmask = tif.asarray().astype(np.uint32)
    elif isinstance(pxmask, np.ndarray):
        pxmask = pxmask.astype(np.uint32)

    with h5py.File(master_name, 'w') as fh:

        detector = fh.create_group('/entry/instrument/detector')
        detectorSpecific = detector.create_group('detectorSpecific')
        data = fh.create_group('/entry/data')

        if pxmask is not None:
            pxmask[pxmask != 0] = 1
            detectorSpecific['pixel_mask'] = pxmask

        nimg = 0

        for ii, fn in enumerate(imgfiles):

            if new_scheme:
                nimg = 0
                with h5py.File(fn, 'r') as img:
                    for _, it in img[data_path].items():
                        target = it.name + '/' + stack
                        data['data_{:06d}'.format(ii + 1)] = h5py.ExternalLink(fn, target)
                        nimg += img[target].shape[0]
            else:
                data['data_{:06d}'.format(ii + 1)] = h5py.ExternalLink(fn, data_path)

            with h5py.File(fn, 'r') as img:
                detectorIn = img['/entry/instrument/detector']
                nimg += detectorIn['data'].shape[0]

                if include_meta and ii == 0:
                    detector['description'] = np.string_('Lambda on Tecnai F20 (Eiger format)')
                    detector['bit_depth_image'] = detectorIn['bit_depth_readout'][:]
                    detectorSpecific['x_pixels_in_detector'] = img[data_path].shape[2]
                    detectorSpecific['y_pixels_in_detector'] = img[data_path].shape[1]
                    detectorSpecific['saturation_value'] = detectorIn['saturation_value'][:]
                    detector['sensor_thickness'] = detectorIn['sensor_thickness'][:] / 1e6
                    detector['x_pixel_size'] = detectorIn['x_pixel_size'][:] / 1e6
                    detector['y_pixel_size'] = detectorIn['y_pixel_size'][:] / 1e6
                    detector['beam_center_x'] = beam_center[0]
                    detector['beam_center_y'] = beam_center[1]
                    detector['count_time'] = detectorIn['count_time'][:] / 1000.
                    detector['frame_time'] = detectorIn['collection/shutter_time'][:] / 1000.
                    detector['detector_distance'] = distance
                fh['/entry/instrument/beam/incident_wavelength'] = wavelength

        detectorSpecific['nimages'] = nimg

    print('Wrote master file {}'.format(master_name))

    return nimg


def sum_movies(files, frames, func=np.sum, suffix='_summed', keep_meta=True):
    """
    Straightforward function to read a Lambda .nxs file, group the frame stack into sub-stacks, and apply an
    aggregation function on the substacks (by default np.sum). Very useful for "serial movie"-type acquisitions.
    :param files: List of file names or file pattern to work on
    :param frames: Number of frames in each sub-stack
    :param keep_meta: Keep the non-image-data in the file TODO: also adjust some of the parameters
    :param func: Aggregation function to apply on the sub-stacks
    :param suffix: File suffix for the aggregated files
    :return: List of created file names
    """
    if isinstance(files, str):
        imgfiles = glob(files)
        imgfiles.sort()
    elif isinstance(files, list) or isinstance(files, tuple):
        imgfiles = files
    else:
        raise ValueError('files must be a file pattern or a list of files!')

    fns2 = []

    for fn in imgfiles:
        fn2 = fn[:-4] + suffix + '.nxs'
        print('Opening {} -> {}'.format(fn, fn2))

        with h5py.File(fn,'r') as fh:
            grp = fh['/entry/instrument/detector']
            imgs = da.from_array(grp['data'], (frames, 516, 1556))
            simgs = imgs.map_blocks(func, axis=0, dtype=imgs.dtype, chunks=(1, 516, 1556))
            da.to_hdf5(fn2, '/entry/instrument/detector/data', simgs,
                       compression=grp['data'].compression, compression_opts=grp['data'].compression_opts, shuffle=True)
        if keep_meta:
            copy_meta(fn, fn2)

        fns2.append(fn2)

    return fns2


def save_lambda_img(img, base_fname='lambda_image', formats=('nxs',),
                    pixel_size=None, meta=None, make_average=False,
                    bigtiff=False, compression=5, **kwargs):
    """
    Save an image (or stack) to a multitude of useful file formats
    :param img: input image as numpy array
    :param base_fname: filename for output without extension
    :param formats: list of formats. can be h5, nxs, tif, cbf, mrc(s)
    :param pixel_size: pixel size in m
    :param meta: metadata to be written into h5, nxs, tif, and partially cbf
    :param make_average: also write a second file, containing an average of the stack
    :param kwargs: will be fed to make_header_string function (for CBF export)
    :return:
    """

    if pixel_size is None:
        pixel_size = 1

    if not (isinstance(formats,list) or isinstance(formats,tuple)):
        formats = (formats,)

    for f in formats:

        fn_out = base_fname + '.' + f

        try:

            if f.lower() == 'tif' or f.lower() == 'tiff':
                with tifffile.TiffWriter(fn_out, bigtiff=bigtiff) as tif:
                    tif.save(img, resolution=(1e2 * pixel_size, 1e2 * pixel_size), compress=compression)
                print('Wrote TIF file: ' + fn_out)

            elif f.lower() == 'mrc' or f.lower() == 'mrcs':
                raise NotImplementedError('MRC does not work at the moment. Sorry.')


            elif f.lower() == 'cbf':
                raise NotImplementedError('CBF does not work at the moment. Sorry.')


            else:
                raise ValueError('Formats can be h5, nxs, tif, or mrc(s)! Use eiger2cbf to make CBF files from nxs.')

        except IOError as err:
            print('Unable to write file: ' + fn_out)
            raise(err)


    if make_average:
        if img.ndim == 3:
            imgs_avg = img.mean(axis = 0, keepdims=False)
            save_lambda_img(imgs_avg, base_fname + '_avg',
                            formats, pixel_size, meta, bigtiff=False,
                            compression=compression, make_average=False, **kwargs)
        else:
            # This does not really make sense (just creates a copy), but could be helpful sometimes
            warn('Requested average image on single-image stack. Totally pointless, but I will still write it.')
            save_lambda_img(img.astype(np.float64), base_fname + '_avg',
                            formats, pixel_size, meta, bigtiff=False,
                            compression=compression, make_average=False, **kwargs)


def read_nxds_spots(filename='SPOT.nXDS', merge_into=None):
    event = -1
    skipnext = True
    linedat = []
    pattern = re.compile('(\d+)\s+')
    for ln, l in enumerate(open(filename,'r')):
        if skipnext:
            skipnext = False
            continue

        nr = pattern.match(l)
        if nr is not None:
            event = int(nr.group(0)) - 1 #nXDS is 1-based
            skipnext = True
            continue

        linedat.append('{} {}'.format(l.strip(), event))

    nxds_peaks = pd.read_csv(StringIO('\n'.join(linedat)), delim_whitespace=True, header=None,
                       names=['Panel','fs/px', 'ss/px', 'Intensity', 'h', 'k', 'l', 'Event']
                       )

    nxds_peaks['Indexer'] = 'nXDS'

    if merge_into is not None:

        merge_into = merge_into.drop(merge_into.columns.intersection(['h', 'k', 'l']), axis=1)

        merge_into['Pos'] = merge_into.groupby('Event').cumcount()
        nxds_peaks['Pos'] = nxds_peaks.groupby('Event').cumcount()
        nxds_peaks = merge_into.merge(nxds_peaks.loc[:, ['Event', 'Pos', 'h', 'k', 'l']],
                                      on=['Event', 'Pos'], how='left', suffixes=('_nXDS', '')).\
                                    drop('Pos', axis=1)

        nxds_peaks[['h', 'k', 'l']] = nxds_peaks[['h', 'k', 'l']].fillna(0).apply(lambda x: x.astype(int))

    return nxds_peaks


def make_eiger_h5(filename, master_name='master.h5', pxmask=None,
                   wavelength=0.025, distance=1.588, beam_center=(778, 308),
                   data_path='/entry/data', stack='centered', legacy=False):
    """
    Mostly intended for nXDS
    :param filename:
    :param master_name:
    :param pxmask:
    :param wavelength:
    :param distance:
    :param beam_center:
    :param data_path:
    :param stack:
    :param legacy:
    :return:
    """

    if pxmask is None:
        pass
    elif isinstance(pxmask, str):
        with tifffile.TiffFile(pxmask) as tif:
            pxmask = tif.asarray().astype(np.uint32)
    elif isinstance(pxmask, np.ndarray):
        pxmask = pxmask.astype(np.uint32)

    with h5py.File(master_name, 'w') as fh, h5py.File(filename, 'r') as img:

        detector = fh.create_group('/entry/instrument/detector')
        detectorSpecific = detector.create_group('detectorSpecific')
        data = fh.create_group('/entry/data')

        if pxmask is not None:
            pxmask[pxmask != 0] = 1
            detectorSpecific['pixel_mask'] = pxmask

        nimg = 0

        for ii, (_, it) in enumerate(img[data_path].items()):
            target = it.name + '/' + stack
            data['data_{:06d}'.format(ii + 1)] = h5py.ExternalLink(filename, target)
            nimg += img[target].shape[0]
            if (pxmask is None) and (ii == 0):
                pxmask = np.zeros(img[target].shape[1:3], dtype=np.uint32)

        fh['/entry/instrument/beam/incident_wavelength'] = wavelength

        detectorSpecific['nimages'] = nimg
        detectorSpecific['ntrigger'] = nimg
        detectorSpecific['pixel_mask'] = pxmask
        detector['detector_distance'] = distance
        detector['x_pixel_size'] = 55e-6
        detector['y_pixel_size'] = 55e-6

    print('Wrote master file {}'.format(master_name))

    return nimg


def load_lambda_img(file_or_ds, range=None, use_dask=False, grp_name='entry/instrument/detector', ds_name='data'):
    """Load a Lambda image stack from raw .nxs HDF5 file. The file can be supplied as either file name, or h5py file
    handle, or h5py data set. Use load_meta to get the metadata."""

    if isinstance(file_or_ds, str):
        ds = h5py.File(file_or_ds, 'r')[grp_name + '/' + ds_name]

    elif isinstance(file_or_ds, h5py.File):
        ds = file_or_ds[grp_name + '/' + ds_name]

    elif isinstance(file_or_ds, h5py.Dataset):
        ds = file_or_ds

    else:
        raise ValueError('file_or_ds must be a file name, a h5py.File, or a h5py.Dataset')

    if use_dask:
        imgs = da.from_array(ds, ds.chunks)
        if range:
            warn('Range parameter is ignored when using Dask!')
    else:
        if range is None:
            imgs = ds[:]
        else:
            imgs = ds[range[0]:range[1],:,:]

    return imgs


def copy_meta(fn_from, fn_to, base_group='/entry/instrument/detector', exclude=('data',), shallow=False):
    """
    Copy sub-tree of h5/nxs file to another one. Typically pretty useful to copy over detector data from nxs files.
    Data in the new file are not overwritten.
    :param fn_from: source file name
    :param fn_to: target file name
    :param base_group: HDF5 path of group to be copied
    :param exclude: list of data fields to exclude. Note: the exclusion only applies to the first level
    of objects below base_group
    :param shallow: only use one level of contents
    """
    print('copy_meta is deprecated. Use copy_h5 instead!')
    fh_to = h5py.File(fn_to)
    fh_from = h5py.File(fn_from, mode='r')
    for k in fh_from[base_group].keys():
        if (k not in exclude) and (k not in fh_to[base_group]) :
            #try:
            fh_from.copy(base_group + '/' + k, fh_to[base_group], shallow=shallow)
            #except ValueError as err:
            #print('Key {} exists'.format(k))
    fh_to.close()
    fh_from.close()