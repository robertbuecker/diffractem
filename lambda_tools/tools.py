# Tool functions to process and convert images from the Lambda detector.

import dask.array as da
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.convolution import Gaussian2DKernel, convolve
import os

from . import gap_pixels
from .io import *
from lambda_tools.legacy import save_lambda_img
from .proc2d import correct_dead_pixels


def diff_plot(filename, idcs, setname='centered', beamdiam=100e-9,
              rings=(10, 5, 2.5), radii=(3, 4, 6), show_map=True, show_peaks=True, show_predict=False,
              figsize=(15, 10), dpi=300, cutoff=99.5,
              width=616, xoff=0, yoff=0, ellipticity=0,
              map_px=None, clen=None, det_px=None, wavelength=None,
              stacks=None, shots=None, peaks=None, predict=None,
              base_path='/%/data', map_path='/%/map/image', results_path='/%/results',
              pre_compute=True, store_to=None, **kwargs):
    """

    """

    if shots is None:
        shots = get_meta_lists(filename, base_path, ['shots'])['shots']
    if show_peaks and peaks is None:
        peaks = get_meta_lists(filename, results_path, ['peaks'])['peaks']
    if show_predict and predict is None:
        predict = get_meta_lists(filename, results_path, ['predict'])['predict']
    if stacks is None:
        stacks = get_data_stacks(filename, base_path, [setname])

    # TODO: replace all the following defaults by proper reading from NeXus and assigning to shots
    shotsel = shots.loc[idcs, :]

    if map_px is None:
        map_px = shotsel['map_px'] = 17e-9
    else:
        shotsel['map_px'] = map_px
    if clen is None:
        shotsel['clen'] = 1.57
    else:
        shotsel['clen'] = clen
    if det_px is None:
        shotsel['det_px'] = 55e-6
    else:
        shotsel['det_px'] = det_px
    if wavelength is None:
        shotsel['wavelength'] = 2.5e-12
    else:
        shotsel['wavelength'] = wavelength

    shotsel['recpx'] = shotsel['wavelength'] / (shotsel['det_px'] / shotsel['clen']) * 1e10

    imgs = stacks[setname][shotsel.index.values, ...]
    if pre_compute:
        imgs = imgs.compute()

    if show_map:
        map_path = map_path.replace('%', 'entry')
        map_imgs = meta_from_nxs(list(shotsel['file'].unique()), map_path)[map_path]

    figs = []

    for ii, ((idx, shot), img) in enumerate(zip(shotsel.iterrows(), imgs)):

        if not pre_compute:
            img = img.compute()

        figh = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        figs.append(figh)
        #figh = figs[ii]

        img_ax = figh.add_axes([0, 0, 0.66, 0.95])
        img_ax.imshow(img, vmin=0, vmax=np.quantile(img, cutoff/100), cmap='gray', label='diff')

        if show_peaks:
            coords = peaks.loc[peaks['serial'] == idx, :]
        else:
            coords = pd.DataFrame()

        if show_predict:
            pred_coords = peaks.loc[predict['serial'] == idx, :]
        else:
            pred_coords = pd.DataFrame()

        img_ax.set_xlim((778 - width/2, 778 + width/2))
        try:
            img_ax.set_title(
                'Set: {}, Shot: {}, Region: {}, Run: {}, Frame: {} \n (#{} in file: {}) PEAKS: {}'.format(shot['subset'],
                                                                                                      idx,
                                                                                                      shot['region'],
                                                                                                      shot['run'],
                                                                                                      shot['frame'],
                                                                                                      shot['shot'],
                                                                                                      shot['file'],
                                                                                                      len(coords), 3))
        except:
            'Shot {}: (file: {}) PEAKS: {}'.format(shot['subset'], shot['file'], len(coords), 3)

        #print(shot['recpx'])
        for res in rings:
            img_ax.add_artist(mpl.patches.Ellipse((img.shape[1] / 2 + xoff, img.shape[0] / 2 + yoff),
                                             width=2*(shot['recpx'] / res), height=2*(shot['recpx'] / res * (1+ellipticity)), edgecolor='y', fill=False))
            img_ax.text(img.shape[1] / 2 + shot['recpx'] / res / 1.4, img.shape[0] / 2 - shot['recpx'] / res / 1.4, '{} A'.format(res),
                    color='y')

        for _, c in coords.iterrows():
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[0], fill=True, color='r', alpha=0.15))
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[1], fill=False, color='y', alpha=0.2))
            img_ax.add_artist(plt.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[2], fill=False, color='y', alpha=0.3))

        for _, c in pred_coords.iterrows():
            img_ax.add_artist(plt.Rectangle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             width=radii[-1], height=radii[-1], fill=False, color='b'))

        img_ax.axis('off')

        if not show_map:
            continue

        #map_px = shotsel['map_px']
        #print(map_px)

        map_ax = figh.add_axes([0.6, 0.5, 0.45, 0.45])
        feat_ax = figh.add_axes([0.6, 0, 0.45, 0.45])

        map_ax.imshow(map_imgs[shot['file']], cmap='gray')
        map_ax.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), facecolor='r'))
        map_ax.add_artist(AnchoredSizeBar(map_ax.transData, 5e-6 / map_px, '5 um', 'lower right'))
        map_ax.axis('off')

        feat_ax.imshow(map_imgs[shot['file']], cmap='gray')
        #feat_ax.add_artist(AnchoredSizeBar(feat_ax.transData, 0.1e-6 / map_px, '100 nm', 'lower right'))
        feat_ax.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), radius=beamdiam/2/map_px, color='r', fill=False))
        if not np.isnan(shot['crystal_x']):
            feat_ax.set_xlim(shot['crystal_x'] + np.array([-20, 20]))
            feat_ax.set_ylim(shot['crystal_y'] + np.array([-20, 20]))
        else:
            feat_ax.set_xlim(shot['pos_x'] + np.array([-20, 20]))
            feat_ax.set_ylim(shot['pos_y'] + np.array([-20, 20]))
        feat_ax.axis('off')

        if store_to is not None:
            plt.savefig('{}/{}_{:04d}'.format(store_to, filename.rsplit('.', 1)[0].rsplit('/', 1)[-1], idx))
            plt.close(plt.gcf())

    return figs


def region_plot(file_name, regions=None, crystal_pos=True, peak_ct=True, beamdiam=100e-9, scanpx=2e-8, figsize=(10, 10),
                **kwargs):
    meta = get_meta_lists(file_name)

    cmap = plt.cm.jet
    fhs = []

    if regions is None:
        regions = meta['shots']['region'].drop_duplicates().values

    if not hasattr(regions, '__iter__'):
        regions = (regions,)

    for reg in regions:

        shots = meta['shots'].loc[meta['shots']['region'] == reg, :]

        if not len(shots):
            print('Region {} does not exist. Skipping.'.format(reg))
            continue

        shot = shots.iloc[0, :]

        fh = plt.figure(figsize=figsize, **kwargs)
        fhs.append(fh)

        ax = plt.axes()
        ax.set_title('Set: {}, Region: {}, Run: {}, # Crystals: {}'.format(shot['subset'], shot['region'], shot['run'],
                                                                           shots['crystal_id'].max()))

        stem = get_meta_array(file_name, 'stem', shot)
        ax.imshow(stem, cmap='gray')

        if 'acqdata' in meta.keys():
            acqdata = meta['acquisition_data'].loc[shot['file']]
            pxs = float(acqdata['Scanning_Pixel_size_x'])
        else:
            pxs = scanpx * 1e9

        if crystal_pos and peak_ct:
            norm = int(shots['peak_count'].quantile(0.99))

            def ncmap(x):
                return cmap(x / norm)

            for idx, cr in shots.loc[:, ['crystal_x', 'crystal_y', 'peak_count']].drop_duplicates().iterrows():
                ax.add_artist(plt.Circle((cr['crystal_x'], cr['crystal_y']), radius=beamdiam * 1e9 / 2 / pxs,
                                         facecolor=ncmap(cr['peak_count']), alpha=1))
            # some gymnastics to get a colorbar
            Z = [[0, 0], [0, 0]]
            levels = range(0, norm, 1)
            CS3 = plt.contourf(Z, levels, cmap=plt.cm.jet)
            plt.colorbar(CS3, fraction=0.046, pad=0.04)
            del (CS3)

        elif crystal_pos:
            for idx, cr in shots.loc[:, ['crystal_x', 'crystal_y']].drop_duplicates().iterrows():
                ax.add_artist(
                    plt.Circle((cr['crystal_x'], cr['crystal_y']), radius=beamdiam * 1e9 / 2 / pxs, facecolor='r',
                               alpha=1))

        ax.add_artist(AnchoredSizeBar(ax.transData, 5000 / pxs, '5 um', 'lower right', pad=0.3, size_vertical=1))
        ax.axis('off')

        return fhs



def make_reference(reference_filename, output_base_fn=None, ref_smooth_range=None,
                   thr_rel_var=0.2, thr_mean=0.2, gap_factor=1, save_stat_imgs=False):
    """Calculates reference data for dead pixels AND flatfield (='gain'='sensitivity'). Returns the boolean dead pixel
    array, and a floating-point flatfield (normalized to median intensity 1)."""

    imgs = da.from_array(h5py.File(reference_filename)['entry/instrument/detector/data'], (100, 516, 1556))
    (mimg, vimg) = da.compute(imgs.mean(axis=0), imgs.var(axis=0))

    # Correct for sensor gaps.
    gap = gap_pixels()
    mimg= mimg.copy()
    vimg = vimg.copy()
    mimg[gap] = mimg[gap]*gap_factor
    vimg[gap] = vimg[gap]*gap_factor*3.2

    # Make mask
    zeropix = mimg == 0
    mimg[mimg == 0] = mimg.mean()

    vom = vimg / mimg
    vom = vom / np.median(vom)
    mimg = mimg / np.median(mimg)

    pxmask = zeropix + (abs(vom - 1) > thr_rel_var) + (abs(mimg - 1) > thr_mean)

    # now calculate the reference, using the corrected mean image
    reference = correct_dead_pixels(mimg, pxmask, interp_range=4)

    if not ref_smooth_range is None:
        reference = convolve(reference, Gaussian2DKernel(ref_smooth_range),
                             boundary='extend', nan_treatment='interpolate')

    # finally undo the gap correction
    reference[gap] = reference[gap]/gap_factor

    reference = reference/np.nanmedian(reference)

    if not output_base_fn is None:
        save_lambda_img(reference.astype(np.float32), output_base_fn + '_reference', formats=('tif', ))
        save_lambda_img(255 * pxmask.astype(np.uint8), output_base_fn + '_pxmask', formats='tif')
        if save_stat_imgs:
            save_lambda_img(mimg.astype(np.float32), output_base_fn + '_mimg', formats=('tif', ))
            save_lambda_img(vom.astype(np.float32), output_base_fn + '_vom', formats=('tif', ))

    return pxmask, reference

# now some functions for mangling with scan lists...

def quantize_y_scan(shots, maxdev=1, min_rows=30, max_rows=500, inc=10, ycol='pos_y', ycol_to=None, xcol='pos_x'):
    """
    Reads a DataFrame containing scan points (in columns xcol and ycol), and quantizes the y positions (scan rows) to
    a reduced number of discrete values, keeping the deviation of the quantized rows to the actual y positions
    below a specified value. The quantized y positions are determined by K-means clustering and unequally spaced.
    :param shots: initial scan list
    :param maxdev: maximum mean standard deviation of row positions from point y coordinates in pixels
    :param min_rows: minimum number of quantized scan rows. Don't set to something unreasonably low, otherwise takes long
    :param max_rows: maximum number of quantized scan rows
    :param inc: step size for determining row number
    :param ycol: column of initial y positions in shots
    :param ycol_to: column of final y row positions in return data frame. If None, overwrite initial y positions
    :param xcol: column of x positions. Required for final sorting.
    :return: scan list with quantized y (row) (positions)
    """

    from sklearn.cluster import KMeans
    if ycol_to is None:
        ycol_to = ycol
    rows = min_rows
    while True:
        shots = shots.copy()
        kmf = KMeans(n_clusters=rows).fit(shots[ycol].values.reshape(-1, 1))
        ysc = kmf.cluster_centers_[kmf.labels_].squeeze()
        shots['y_dev'] = shots[ycol] - ysc
        if np.sqrt(kmf.inertia_/len(shots)) <= maxdev:
            print('Reached y deviation goal with {} scan rows.'.format(rows))
            shots[ycol_to] = ysc
            shots.sort_values(by=[ycol, xcol], inplace=True)
            return shots.reset_index(drop=True)
        rows += inc


def set_frames(shots, frames=1):
    """
    Adds additional frames to each scan position by repeating each line, and adding/setting a frame column
    :param shots: initial scan list. Each scan points must have a unique index, otherwise behavior may be funny.
    :param frames: number of frames per scan position
    :return: scan list with many frames per position
    """

    if frames > 1:
        shl_rep = shots.loc[shots.index.repeat(frames), :].copy()
        shl_rep['frame'] = np.hstack([np.arange(frames)] * len(shots))
    else:
        shl_rep = shots
        shl_rep['frame'] = 1
    return shl_rep


def insert_init(shots, predist=100, dxmax=200, xcol='pos_x', initpoints=1):
    """
    Insert initialization frames into scan list, to mitigate hysteresis and beam tilt streaking when scanning along x.
    Works by inserting a single frame each time the x coordinate decreases (beam moves left) or increases by more
    than dxmax (beam moves too quickly). The initialization frame is taken to the left of the position after the jump by
    predist pixels. Its crystal_id and frame columns are set to -1.
    :param shots: initial scan list. Note: if you want to have multiple frames, you should always first run set_frames
    :param predist: distance of the initialization shot from the actual image along x
    :param dxmax: maximum allowed jump size (in pixels) to the right.
    :param xcol: name of x position column
    :param initpoints: number of initialization points added
    :return: scan list with inserted additional points
    """

    def add_init(sh1):
        initline = sh1.iloc[:initpoints, :].copy()
        initline['crystal_id'] = -1
        initline['frame'] = -1
        if predist is not None:
            initline[xcol] = initline[xcol] - predist
        else:
            initline[xcol] = 0
        return initline.append(sh1)

    dx = shots[xcol].diff()
    grps = shots.groupby(by=((dx < 0) | (dx > dxmax)).astype(int).cumsum())
    return grps.apply(add_init).reset_index(drop=True)


def call_indexamajig(input, geometry, output='im_out.stream', cell=None, im_params=None, index_params=None,
                     procs=40, exc='indexamajig', **kwargs):

    '''Generates an indexamajig command from a dictionary of indexamajig parameters, a exc dictionary of files names and core number, and an indexer dictionary

    e.g.

    im_params = {'min-res': 10, 'max-res': 300, 'min-peaks': 0,
                      'int-radius': '3,4,6', 'min-snr': 4, 'threshold': 0,
                      'min-pix-count': 2, 'max-pix-count':100,
                      'peaks': 'peakfinder8', 'fix-profile-radius': 0.1e9,
                      'indexing': 'none', 'push-res': 2, 'no-cell-combinations': None,
                      'integration': 'rings-rescut','no-refine': None,
                      'no-non-hits-in-stream': None, 'no-retry': None, 'no-check-peaks': None} #'local-bg-radius': False,

    index_params ={'pinkIndexer-considered-peaks-count': 4,
             'pinkIndexer-angle-resolution': 4,
             'pinkIndexer-refinement-type': 0,
             'pinkIndexer-thread-count': 1,
             'pinkIndexer-tolerance': 0.10}
             '''


    exc_dic = {'g': geometry, 'i': input, 'o': output, 'j': procs}

    for k, v in exc_dic.items():
        exc += f' -{k} {v}'

    if cell is not None:
        exc += f' -p {cell}'

    for kk, vv in im_params.items():
        if vv is not None:
            exc += f' --{kk}={vv}'
        else:
            exc += f' --{kk}'

    # If the indexer dictionary is not empty
    if index_params:
        for kkk, vvv in index_params.items():
            if vvv is not None:
                exc += f' --{kkk}={vvv}'
            else:
                exc += f' --{kkk}'

    return exc


def dict2file(file_name, file_dic, header=None):

    fid = open(file_name, 'w')  # Open file

    if header is not None:
        fid.write(header)  # Header
        fid.write("\n\n")

    for k, v in file_dic.items():
        fid.write("{} = {}".format(k, v))
        fid.write("\n")

    fid.close()  # Close file


def make_geometry(parameters, file_name=None):
    par = {'photon_energy': 495937,
           'adu_per_photon': 2,
           'clen': 1.587900,
           'res': 18181.8181818181818181,
           'mask': '/entry/data/%/pxmask_centered',
           'mask_good': '0x01',
           'mask_bad': '0x00',
           'data': '/entry/data/%/centered',
           'dim0': '%',
           'dim1': 'ss',
           'dim2': 'fs',
           'p0/min_ss': 0,
           'p0/max_ss': 615,
           'p0/min_fs': 0,
           'p0/max_fs': 1555,
           'p0/corner_y': -308,
           'p0/corner_x': -778,
           'p0/fs': '+x',
           'p0/ss': '+y'}

    par.update(parameters)

    if file_name is not None:
        dict2file(file_name, par, header=';Auto-generated Lambda detector file')

    return par

