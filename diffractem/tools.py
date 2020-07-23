# Miscellaneous supporting tool functions

from build.lib.diffractem.pre_process import PreProcOpts
from astropy.convolution import Gaussian2DKernel, convolve

from . import gap_pixels
from .stream_parser import StreamParser
# from .dataset import Dataset
from .io import *
from .pre_proc_opts import PreProcOpts
from .proc2d import correct_dead_pixels
from typing import Iterable, Tuple, Union, Optional
import os
import shutil
from tifffile import imread, imsave
import hashlib
import pandas as pd
from warnings import warn
import numpy as np
import subprocess

def dataframe_hash(df: pd.DataFrame, string: bool = True, signed: bool = True) -> pd.Series:
    """
    Generate int32 (or uint32 if signed=False) hashes from a pandas data frame and return as a pandas Series.
    The hash is generated with md5 from a whitespace-delimited string representation from each data frame row, 
    such as e.g.:
    "data/file_a.h5 entry//0" (so similar to CrystFEL's list syntax).
    """
    
    if string:
        hfunc = lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()
    else:    
        hfunc = lambda s: int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)
    
    str_series = None
    for _, s in df.astype(str).items():
        str_series = s if str_series is None else str_series + ' ' + s
        
    hashes = str_series.apply(hfunc)
    
    if len(hashes.unique()) != len(hashes):
        warn(f'dataframe hash for cols {", ".join(df.columns)} is non-unique! This will likely cause trouble downstream.', 
             RuntimeWarning)

    if string:
        return hashes
    else:
        return (hashes - 2**32//2).astype(np.int32) if signed else hashes.astype(np.uint32)


def strip_file_path(df: pd.DataFrame, add_folder=False):
    splt = df.file.str.rsplit('/', 1, True)
    if add_folder:
        if splt.shape[1] == 2:
            return df.assign(file=splt.iloc[:, -1], folder=splt.iloc[:, 0])
        else:
            return df.assign(file=splt.iloc[:, -1], folder='.')
    else:
        return df.assign(file=splt.iloc[:, -1])


def make_reference(reference_filename, output_base_fn=None, ref_smooth_range=None,
                   thr_rel_var=0.2, thr_mean=0.2, gap_factor=1, save_stat_imgs=False):
    """Calculates reference data for dead pixels AND flatfield (='gain'='sensitivity'). Returns the boolean dead pixel
    array, and a floating-point flatfield (normalized to median intensity 1)."""

    imgs = da.from_array(h5py.File(reference_filename)['entry/instrument/detector/data'], (100, 516, 1556))
    (mimg, vimg) = da.compute(imgs.mean(axis=0), imgs.var(axis=0))

    # Correct for sensor gaps.
    gap = gap_pixels()
    mimg = mimg.copy()
    vimg = vimg.copy()
    mimg[gap] = mimg[gap] * gap_factor
    #TODO this needs some more consideration. Neither 3 nor sqrt(3) really work.
    vimg[gap] = vimg[gap] * gap_factor * 1.8 #3.2

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
    reference[gap] = reference[gap] / gap_factor

    reference = reference / np.nanmedian(reference)

    if not output_base_fn is None:
        imsave(output_base_fn + '_reference.tif', reference.astype(np.float32))
        imsave(output_base_fn + '_pxmask.tif', 255 * pxmask.astype(np.uint8))
        if save_stat_imgs:
            imsave(output_base_fn + '_mimg.tif', mimg.astype(np.float32))
            imsave(output_base_fn + '_vom.tif', vom.astype(np.float32))

    return pxmask, reference


# now some functions for mangling with scan lists...

def quantize_y_scan(shots, maxdev=1, min_rows=30, max_rows=500, 
    inc=10, ycol='pos_y', ycol_to=None, xcol='pos_x', adaptive=True):
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
    dev = -1
    while True:
        shots = shots.copy()
        kmf = KMeans(n_clusters=rows).fit(shots[ycol].values.reshape(-1, 1))
        ysc = kmf.cluster_centers_[kmf.labels_].squeeze()
        shots['y_dev'] = shots[ycol] - ysc
        dev0 = dev
        dev = np.sqrt(kmf.inertia_ / len(shots))
        print(f'Scan point quantization:, {rows} rows, {dev:0.3f} deviation (reduced by {(dev0-dev)/dev0:0.3}).')
        if dev <= maxdev:
            print('Reached y deviation goal with {} scan rows.'.format(rows))
            shots[ycol_to] = ysc
            shots.sort_values(by=[ycol, xcol], inplace=True)
            return shots.reset_index(drop=True)
        if adaptive and (dev0 > 0):
            inc = int(inc/((dev0-dev)/dev0))
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


def make_command(program, arguments=None, params=None, opts=None, *args, **kwargs):
    exc = program
    if arguments is None:
        arguments = []
    if not isinstance(arguments, (list, tuple)):
        arguments = [arguments, ]
    arguments.extend(args)
    if arguments is not None:
        for a in arguments:
            exc += f' {a}'

    if params is not None:
        for p, v in params.items():
            exc += f' -{p} {v}'

    if opts is None:
        opts = {}
    opts.update(kwargs)
    opts = {k.replace('_', '-'): v for k, v  in opts.items()}
    for o, v in opts.items():
        if (v is not None) and not isinstance(v, bool):
            exc += f' --{o}={v}'
        elif (v is None) or v:
            exc += f' --{o}'

    return exc


def call_partialator(input, symmetry, output='im_out.stream', model='unity', iterations=1, opts=None,
                     procs=40, exc='partialator'):
    params = {'y': symmetry, 'i': input, 'o': output, 'j': procs, 'n': iterations, 'm': model}

    return make_command(exc, None, params, opts=opts)


def call_indexamajig(input, geometry, output='im_out.stream', cell: Optional[str] = None,
                     im_params: Optional[dict] = None, index_params: Optional[dict] = None,
                     procs: Optional[int] = None, exc='indexamajig', copy_fields: list = (), **kwargs):
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

    exc_dic = {'g': geometry, 'i': input, 'o': output, 'j': os.cpu_count() if procs is None else procs}
    if cell is not None:
        exc_dic['p'] = cell

    options = dict(im_params if im_params is not None else {}, **({} if index_params is None else index_params), **kwargs)
    options = {k.replace('_', '-'): v for k, v  in options.items()}

    cmd = make_command(exc, None, params=exc_dic, opts=options)
    
    for f in copy_fields:
        cmd += f' --copy-hdf5-field={f}'

    return cmd


def dict2file(file_name, file_dic, header=None):
    fid = open(file_name, 'w')  # Open file

    if header is not None:
        fid.write(header)  # Header
        fid.write("\n\n")

    for k, v in file_dic.items():
        fid.write("{} = {}".format(k, v))
        fid.write("\n")

    fid.close()  # Close file


def make_geometry(opts: PreProcOpts, file_name=None, image_name='corrected',
                  xsize: Optional[int] = None, ysize: Optional[int] = None, **kwargs):
    
    xsz = opts.xsize if xsize is None else xsize
    ysz = opts.ysize if ysize is None else ysize
    
    # get ellipticity correction
    c, s = np.cos(opts.ellipse_angle*np.pi/180), np.sin(opts.ellipse_angle*np.pi/180)
    R = np.array([[c, -s], [s, c]])
    # note that the ellipse values are _inverted_ (x' gets scaled by 1/sqrt(ratio))
    RR = R.T @ ([[opts.ellipse_ratio**(-.5)],[opts.ellipse_ratio**(.5)]] * R)
    X0 = RR @ [[-xsz//2], [-ysz//2]]
    
    par = {'photon_energy': 1.23984197e4 / opts.wavelength,
           'adu_per_photon': 1.9,
           'clen': opts.cam_length,
           'res': 1 / opts.pixel_size,
           'mask': '/mask',
           'mask_file': 'pxmask.h5',
           'mask_good': '0x01',
           'mask_bad': '0x00',
           'data': '/%/data/' + image_name,
           'dim0': '%',
           'dim1': 'ss',
           'dim2': 'fs',
           'p0/min_ss': 0,
           'p0/max_ss': ysz - 1,
           'p0/min_fs': 0,
           'p0/max_fs': xsz - 1,
           'p0/corner_x': X0[0,0],
           'p0/corner_y': X0[1,0],
           'p0/fs': f'{RR[0,0]:+.04f}x {RR[0,1]:+.04f}y',
           'p0/ss': f'{RR[1,0]:+.04f}x {RR[1,1]:+.04f}y'}

    par.update(kwargs)
    
    if file_name is not None:
        dict2file(file_name, par, 
                  header=';Lambda detector file generated by diffractem.\n'
                  f';Ellipticity correction with ratio {opts.ellipse_ratio}, angle {opts.ellipse_angle} deg.')

    return par


def chop_stream(streamfile: str, shots: pd.DataFrame, query='frame == 1', postfix='fr1'):
    """Carve shots with a given frame number from a stream file
    
    Arguments:
        streamfile {str} -- [stream file to open]
        frame {int} -- [frame number to extract]
    """
    stream = StreamParser(streamfile)
    skip = stream.shots[['frame', 'first_line', 'last_line']].merge(shots, on=['file', 'Event']). \
        query(query).sort_values(by='first_line', ascending=False)

    start = list(skip.first_line)
    stop = list(skip.last_line)

    with open(streamfile, 'r') as fh_in, \
            open(streamfile.rsplit('.', 1)[0] + postfix + '.stream', 'w') as fh_out:
        lstart = start.pop()
        lstop = -1
        exclude = False
        for ln, l in enumerate(fh_in):
            if exclude:
                if ln == lstop:
                    exclude = False
                    try:
                        lstart = start.pop()
                    except IndexError:
                        lstart = -1
                        # print('Reached last exclusion')
            else:
                if ln == lstart:
                    exclude = True
                    lstop = stop.pop()
                else:
                    fh_out.write(l)

def analyze_hkl(fn: str, cell: str, point_group: str, foms: Iterable = ('CC', 'CCstar', 'Rsplit'), 
                nshells: int = 10, lowres: float = 35, highres: float = 1.5,
                fn1: Optional[str] = None, fn2: Optional[str] = None, 
                shell_dir: str = 'shell', bin_path: str = '') -> Tuple[dict, pd.DataFrame, str]:
    """Analyze a `hkl`-file triplet as generated by `partialator` (comprising `hkl`, `hkl1, `hkl2`)

    Args:
        fn (str): [description]
        cell (str): [description]
        point_group (str): [description]
        foms (Iterable, optional): [description]. Defaults to ('CC', 'CCstar', 'Rsplit').
        nshells (int, optional): [description]. Defaults to 10.
        lowres (float, optional): [description]. Defaults to 35.
        highres (float, optional): [description]. Defaults to 1.5.
        fn1 (Optional[str], optional): [description]. Defaults to None.
        fn2 (Optional[str], optional): [description]. Defaults to None.
        shell_dir (str, optional): [description]. Defaults to 'shell'.
        bin_path (str, optional): [description]. Defaults to ''.

    Raises:
        cpe: [description]

    Returns:
        [type]: [description]
    """
    
    fnroot = fn.rsplit('/',1)[-1].rsplit('.')[0]

    foms = list(foms)
    
    os.makedirs(shell_dir, exist_ok=True)
    
    if (fn1 is None) or (fn2 is None):
        fn1 = fn + '1'
        fn2 = fn + '2'

    callstr = make_command(os.path.join(bin_path, 'check_hkl'), fn, 
                                 {'y': point_group, 'p': cell},
                                  {'shell-file': f'{shell_dir}/hkl_{fnroot}.dat'}, 
                                 nshells=nshells, lowres=lowres, highres=highres)

    try:
        so = subprocess.check_output(callstr.split(), stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as cpe:
        print(f'check_hkl for file {fnroot} failed because of: ')    
        print(cpe.output.decode())
        print(f'Abandoning.')
        raise cpe

    sd1 = pd.read_csv(f'{shell_dir}/hkl_{fnroot}.dat', delim_whitespace=True, header=None, skiprows=1)
    sd1.columns = ['Center 1/nm', 'nref', 'Possible',  
                                    'Compl', 'Meas', 'Red', 'SNR', 'Std dev', 
                                    'Mean', 'd/A', 'Min 1/nm', 'Max 1/nm']

    idx = 0
    sd2 = None
    for fom in foms:
        callstr = make_command(os.path.join(bin_path, 'compare_hkl'), [fn1, fn2], 
                                 {'y': point_group, 'p': cell},
                                  {'shell-file': f'{shell_dir}/cmp_{fnroot}.dat'},#, 'sigma-cutoff': -4}, 
                                     nshells=nshells, lowres=lowres, highres=highres, fom=fom)

        # print('Running', callstr)
        try:
            so2 = subprocess.check_output(callstr.split(), stderr=subprocess.STDOUT).decode()
            so += so2
        except subprocess.CalledProcessError as cpe:
            print(f'compare_hkl for figure-of-merit {fom} for set {fnroot} failed because of: ')    
            print(cpe.output.decode())
            print('Trying to continue with other FOMs')
            continue

        if idx==0:
            sd2 = pd.read_csv(f'{shell_dir}/cmp_{fnroot}.dat', delim_whitespace=True, 
                                 header=None, skiprows=1)
            sd2 = sd2[[0, 2, 3, 4, 5, 1]]
        else:
            sd2 = pd.concat([sd2, pd.read_csv(f'{shell_dir}/cmp_{fnroot}.dat', delim_whitespace=True, 
                                 header=None, skiprows=1, usecols=[1])], axis=1, ignore_index=True)            
        idx += 1

    if sd2 is None:
        warn('None of the FOMs worked!')
        sd = sd1
    else:
        sd2.columns = ['Center 1/nm', 'nref', 'd/A', 'Min 1/nm', 'Max 1/nm'] + foms
        sd = pd.merge(sd1, sd2, on='Center 1/nm', suffixes=('', '__2'))
        sd.drop([c for c in sd.columns if c.endswith('__2')], axis=1, inplace=True)

    # parse for 'overall quantities'
    overall = {}
    for l in [l for l in so.split('\n') if ('overall' in l.lower()) and ('=' in l)]:
        # print(l)
        sect = l.split('=')
        fld = sect[0].strip().rsplit(' ', 1)[-1]
        overall[fld] = float(sect[1].strip().split(' ')[0])

    return sd, overall, so