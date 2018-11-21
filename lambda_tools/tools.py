# Tool functions to process and convert images from the Lambda detector.

import dask.array as da
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from astropy.convolution import Gaussian2DKernel, convolve

from . import gap_pixels
from .io import save_lambda_img, get_meta_lists, get_data_stacks, get_meta_array
from .proc2d import correct_dead_pixels


def diff_plot(file_name, idcs, setname='centered', ovname='stem', radii=(3, 4, 6), beamdiam=100e-9,
              rings=(10, 5, 2.5), scanpx=20e-9, clen=1.59, figsize=(15, 10), **kwargs):
    """
    Makes a single or multiple nice plots of a diffraction pattern and associated STEM image.
    :param file_name:
    :param idcs: can be a single index, or a range (or anything compatible with a DataFrame .loc)
    :param setname: name of the image set
    :param ovname: name of the overview set
    :param radii: radii of points at the found diffraction peaks
    :param beamdiam: beam diameter (in m) shown in zoomed crystal image
    :param rings: radii of shown reference diffraction rings
    :param clen: camera length the data has been taken with
    :param scanpx: pixel size of overview image. Only used if no acquisition data is stored inside the file.
    :param figsize: figure size
    :param kwargs: other keyword arguments passed to figure command
    :return: list of figure handles
    """

    meta = get_meta_lists(file_name, flat=True)
    imgset = get_data_stacks(file_name, flat=True)[setname]
    shots = meta['shots'].loc[idcs, :]
    recpx = 0.025 / (55e-6 / clen)

    if isinstance(shots, pd.Series):
        shots = pd.DataFrame(shots).T

    fh = []

    for idx, shot in shots.iterrows():

        fhs = plt.figure(figsize=figsize, **kwargs)
        fh.append(fhs)
        ax = plt.axes([0, 0, 0.66, 0.95])
        ax2 = plt.axes([0.6, 0.5, 0.45, 0.45])
        ax3 = plt.axes((0.6, 0, 0.45, 0.45))

        dat = imgset[idx, ...].compute()
        ax.imshow(dat, vmin=0, vmax=np.quantile(dat, 0.995), cmap='gray', label='diff')

        if 'peaks' in meta.keys():
            coords = meta['peaks'].loc[meta['peaks']['Event'] == idx, :]
        else:
            coords = pd.DataFrame()

        ax.set_xlim((778 - 306, 778 + 306))
        ax.set_title(
            'Set: {}, Shot: {}, Region: {}, Run: {}, Frame: {} \n (#{} in file: {}) PEAKS: {}'.format(shot['subset'],
                                                                                                      idx,
                                                                                                      shot['region'],
                                                                                                      shot['run'],
                                                                                                      shot['frame'],
                                                                                                      shot['shot'],
                                                                                                      shot['file'],
                                                                                                      len(coords), 3))

        for res, col in zip(rings, 'yyy'):
            ax.add_artist(mpl.patches.Circle((dat.shape[1] / 2, dat.shape[0] / 2),
                                             radius=recpx / res, edgecolor=col, fill=False))
            ax.text(dat.shape[1] / 2 + recpx / res / 1.4, dat.shape[0] / 2 - recpx / res / 1.4, '{} A'.format(res),
                    color=col)

        for _, c in coords.iterrows():
            ax.add_artist(mpl.patches.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[0], fill=True, color='r', alpha=0.15))
            ax.add_artist(mpl.patches.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[1], fill=False, color='y', alpha=0.2))
            ax.add_artist(mpl.patches.Circle((c['fs/px'] - 0.5, c['ss/px'] - 0.5),
                                             radius=radii[2], fill=False, color='y', alpha=0.3))

        ax.axis('off')

        stem = get_meta_array(file_name, ovname, shot)

        if 'acqdata' in meta.keys():
            pxs = float(meta['acqdata']['Scanning_Pixel_size_x'])
        else:
            pxs = scanpx * 1e9

        ax2.imshow(stem, cmap='gray')
        ax2.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), facecolor='r'))
        ax2.add_artist(AnchoredSizeBar(ax2.transData, 5000 / pxs, '5 um', 'lower right'))
        ax2.axis('off')

        ax3.imshow(stem, cmap='gray')
        if not np.isnan(shot['crystal_x']):
            ax3.set_xlim(shot['crystal_x'] + np.array([-20, 20]))
            ax3.set_ylim(shot['crystal_y'] + np.array([-20, 20]))
        else:
            ax3.set_xlim(shot['pos_x'] + np.array([-20, 20]))
            ax3.set_ylim(shot['pos_y'] + np.array([-20, 20]))
        ax3.add_artist(AnchoredSizeBar(ax3.transData, 100 / pxs, '100 nm', 'lower right'))
        ax3.add_artist(plt.Circle((shot['crystal_x'], shot['crystal_y']), radius=beamdiam*1e9/2/pxs, facecolor='r', alpha=0.2))
        ax3.axis('off')

    return fh


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

def quantize_y_scan(shots, maxdev=1, min_rows=100, max_rows=500, inc=10, ycol='pos_y', ycol_to=None, xcol='pos_x'):
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


def insert_init(shots, predist=100, dxmax=200, xcol='pos_x'):
    """
    Insert initialization frames into scan list, to mitigate hysteresis and beam tilt streaking when scanning along x.
    Works by inserting a single frame each time the x coordinate decreases (beam moves left) or increases by more
    than dxmax (beam moves too quickly). The initialization frame is taken to the left of the position after the jump by
    predist pixels. Its crystal_id and frame columns are set to -1.
    :param shots: initial scan list. Note: if you want to have multiple frames, you should always first run set_frames
    :param predist: distance of the initialization shot from the actual image along x
    :param dxmax: maximum allowed jump size (in pixels) to the right.
    :param xcol: name of x position column
    :return: scan list with inserted additional points
    """

    def add_init(sh1):
        initline = sh1.iloc[:1, :].copy()
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


# DOES NOT WORK YET
def call_crystfel(filename, streamfile='cfout.stream', procs=40, exc='opts/crystfel/bin/indexamajig',
                  geom='lambda_centered_h5.geom', **kwargs):

    raise NotImplementedError('This function does not work yet.')

    with open('cfrun.lst', 'w') as fh:
        fh.write(filename)

    call = [exc, '-g ' + geom, '-j {}'.format(procs), '-i cfrun.lst', '-o ' + streamfile]

    for k, v in kwargs:
        call.append('--{}={}'.format(v))

    import subprocess

    p = subprocess.Popen(call, stdout=subprocess.PIPE)

    while p.poll() is None:
        l = p.stdout.readline()  # This blocks until it receives a newline.
        print(l)

    print(p.stdout.read())
