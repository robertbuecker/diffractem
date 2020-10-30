# Miscellaneous supporting tool functions
from astropy.convolution import Gaussian2DKernel, convolve
from . import gap_pixels
from .stream_parser import StreamParser
# from .dataset import Dataset
from .io import *
from .pre_proc_opts import PreProcOpts
from .proc2d import correct_dead_pixels
from . import gap_pixels, panel_pix
from typing import Iterable, Tuple, Union, Optional
import os
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
                   thr_rel_var=0.2, thr_mean=0.2, gap_factor=1, save_stat_imgs=False, per_panel=False):
    """Calculates reference data for dead pixels AND flatfield (='gain'='sensitivity'). Returns the boolean dead pixel
    array, and a floating-point flatfield (normalized to median intensity 1)."""

    imgs = da.from_array(h5py.File(reference_filename)['entry/instrument/detector/data'])
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
    
    mimg_rel = mimg.copy() / np.median(mimg)
    vom = vimg / mimg

    if per_panel:
        from diffractem import panel_pix
        for pid in range(1,13,1):
            sel = panel_pix(pid, include_gap=False)
            vom[sel] = vom[sel] / np.median(vom[sel])
            mimg[sel] = mimg[sel] / np.median(mimg[sel])
        vom[gap] = vom[gap] / np.median(vom[gap])
        mimg[gap] = mimg[gap] / np.median(mimg[gap])
    else:
        vom = vom / np.median(vom)
        mimg = mimg / np.median(mimg)

    pxmask = zeropix + (abs(vom - 1) > thr_rel_var) + (abs(mimg - 1) > thr_mean)

    # now calculate the reference, using the corrected mean image
    reference = correct_dead_pixels(mimg_rel, pxmask, interp_range=4, edge_mask_x=(100, 50))

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


def call_partialator_simple(input, symmetry, output='im_out.stream', model='unity', iterations=1, opts=None,
                     procs=40, exc='partialator'):
    params = {'y': symmetry, 'i': input, 'o': output, 'j': procs, 'n': iterations, 'm': model}

    return make_command(exc, None, params, opts=opts)


def call_partialator(input: Union[list,str], options: dict, script_name: Optional[str] = 'partialator_run.sh',
                     cache_streams: bool = False, 
                     par_runs: int = 1, slurm: bool = False, split: bool = False,
                     out_dir: str = 'merged', slurm_opts: Optional[dict] = None, 
                     exc='partialator') -> Optional[str]:
    
    if slurm and not cache_streams:
        warn('If making a SLURM script, you usually want to set cache_streams=True to get the paths right on a cluster.')
    
    # looks weird, but required to have input first in the dict:
    _options = {'input': input}
    _options.update(options)
    options = _options
    
    from itertools import product
    changing = {k: v for k, v in options.items() if (isinstance(v, list) or isinstance(v, tuple))}
    settings = pd.DataFrame([dict(zip(changing.keys(), element)) for element in product(*changing.values())])

    full_call = '#!/bin/sh\n'

    # copy stream files to some fast local storage
    if cache_streams:
        full_call += f'mkdir -p {out_dir}\n'
        for fn in input:
            full_call += f'cp {fn} {out_dir}\n'
            if split:
                full_call += f'cp {fn.rsplit(".", 1)[0]}_split.txt {out_dir}\n'

    ii = 0 
        
    for idx, s in settings.iterrows():
        the_par = options.copy()
        the_par.update(dict(s))
        if 'input' in s:
            s['input'] = os.path.basename(the_par['input']).rsplit('.', 1)[0]
        partialator_out = '__'.join([str(v) for k, v in s.items()])
        the_par['output'] = os.path.join('..', partialator_out) + '.hkl'
        the_par['input'] = os.path.join('..', os.path.basename(the_par['input'])) if cache_streams \
                        else os.path.join(os.getcwd(), the_par['input'])   
        if split:
            the_par['custom-split'] = the_par['input'].rsplit('.', 1)[0] + '_split.txt'
        partstr = make_command(exc, params={k: v for k, v in the_par.items() if len(k)==1}, 
                opts={k: v for k, v in the_par.items() if len(k)>1})

        hkldir = os.path.join(out_dir, partialator_out)
        
        settings.loc[idx,'hklfile'] = os.path.join(out_dir, os.path.basename(the_par['output']))

        callstr = f'(mkdir -p {hkldir}/pr-logs; cd {hkldir}; rm -rf pr-logs; {partstr}'

        if slurm:
            callstr += ')'
            slurm_opts['job-name'] = f'"{partialator_out}"'
            callstr = make_command('sbatch', params={k: v for k, v in slurm_opts.items() if len(k)==1}, 
                                        opts={k: v for k, v in slurm_opts.items() if len(k)>1},
                                    ntasks=the_par['j'], wrap=f'"{callstr}"', 
                                        output='partialator.out',
                                        error='partialator.err') + ' \n'
        else:
            # callstr += (f'> {partialator_out}.out 2> {partialator_out}.err ' 
            #             + (') \n' if ((ii + 1) % par_runs) == 0 else ') &\n'))
            callstr += (f' 2> partialator.err > partialator.out ' 
                        + (') \n' if ((ii + 1) % par_runs) == 0 else ') & \n'))

        # print(callstr)
        ii += 1
        
        full_call += callstr
    
    settings.columns = [cn.replace('-', '_') for cn in settings.columns]
    
    if script_name is not None:
        with open(script_name, 'w') as fh:
            fh.write(full_call)
        print('Please run', script_name, 'to start merging.')
        return settings
    else:
        return settings, full_call
    
    
def get_hkl_settings(filenames: Union[list, str], unique_only=False, custom_split=False):
    import re
    settings = []
    hklfiles = glob(filenames) if isinstance(filenames, str) else filenames
    for fn in hklfiles:
        if custom_split and os.path.exists(fn.rsplit('-', 1)[0] + '.hkl'):
            base_fn = fn.rsplit('-', 1)[0] + '.hkl'
            splt_lbl = fn.rsplit('-', 1)[-1].rsplit('.', 1)[0]
        else:
            base_fn = fn
            splt_lbl = ''
        with open(base_fn, 'r') as fh:
            for ln in fh:
                if 'partialator' in ln:
                    s = {'hklfile': fn, 'split_label': splt_lbl}
                    for opt in re.split(' --| -', ln)[1:]:                
                        p = re.split(' |=', opt)
                        if p[0] in ['o', 'output']:
                            continue
                        if p[0] in ['i', 'input']:
                            p[1] = p[1].strip(os.getcwd())
                        s[p[0].replace('-', '_')] = p[1].strip() if len(p) > 1 else True
                    settings.append(s)
                    
    settings = pd.DataFrame(settings).apply(pd.to_numeric, errors='ignore')
    if unique_only:
        nunique = settings.apply(pd.Series.nunique)
        settings.drop(nunique[nunique == 1].index, axis=1, inplace=True)
    return settings
    

def call_indexamajig_slurm(input, geometry, name='idx', cell: Optional[str] = None,
                     im_params: Optional[dict] = None,
                     procs: Optional[int] = None, exc='indexamajig', copy_fields: list = (), 
                     shots_per_run: int = 50, partition: str = 'medium', time: str = '01:59:00', 
                     folder='partitions', tar_file: Optional[str] = None, threads: int = 1,
                     local_bin_dir: Optional[str] = None, **kwargs):
    
    script_name = f'im_run_{name}.sh'
    tar_file = f'{name}.tar.gz'

    cf_call = []
    os.makedirs(folder, exist_ok=True)
    [os.remove(folder+'/'+fn) for fn in os.listdir(folder)]

    from subprocess import run
    local_bin_dir = '' if local_bin_dir is None else local_bin_dir
    run(f'{os.path.join(local_bin_dir,"list_events")} -i {input} -g virtual.geom -o {input}-shots.lst'.split(' '))
    shots = pd.read_csv(f'{input}-shots.lst', delim_whitespace=True, names=['file', 'Event'])
    os.remove(f'{input}-shots.lst')

    if threads > 1:
        # add thread parameters here if needed (e.g. for other indexers)
        im_params['pinkIndexer-thread-count'] = threads

    for ii, grp in shots.groupby(shots.index // shots_per_run):
        jobname = f'{name}_{ii:03d}'
        basename = f'{folder}/' + jobname
        grp[['file', 'Event']].to_csv(basename + '.lst', header=False, index=False, sep=' ')
        callstr = call_indexamajig(basename + '.lst', geometry, basename + '.stream', 
                                cell=cell, im_params=im_params, 
                                procs=procs, exc=exc,
                                no_refls_in_stream=False, serial_start=grp.index[0]+1,
                                copy_fields=copy_fields, **kwargs)
        
        slurmstr = make_command('sbatch', partition=partition, job_name=jobname, 
                                    time=time, nodes=1, ntasks=procs*threads,
                                    output=basename + '.out', error=basename + '.err',
                                    wrap=f"'{callstr}'")
        
        cf_call.append(slurmstr)
        
    with open(script_name, 'w') as fh:
        fh.write('\n'.join(cf_call))
    os.chmod(script_name, os.stat(script_name).st_mode | 0o111)

    if tar_file is not None:
        import tarfile
        with tarfile.open(tar_file, 'w:gz' if tar_file.endswith('.gz') else 'w') as tar:
            tar.add(folder)
            tar.add(geometry)
            tar.add(cell)
            tar.add(script_name)
            for fn in shots['file'].unique():
                tar.add(fn)
        print(f'Wrote self-contained tar file {tar_file}. ' 
              f'Upload to your favorite cluster and extract with: tar -xf {tar_file}')
        
    print(f'Run indexing by calling ./{script_name}')
    
    return tar_file, script_name
    
    
def call_indexamajig(input, geometry, output='im_out.stream', cell: Optional[str] = None,
                     im_params: Optional[dict] = None, index_params: Optional[dict] = None,
                     procs: Optional[int] = None, exc='indexamajig', copy_fields: list = (), 
                     script: Optional[str] = None, **kwargs):
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

    if script:
        with open(script, 'w') as fh:
            fh.write(cmd)
        os.chmod(script, os.stat(script).st_mode | 0o111)
        return None
    else:
        return cmd


def dict2file(file_name, file_dic, header=None):
    
    with open(file_name, 'w') as fid:  # Open file

        if header is not None:
            fid.write(header)  # Header
            fid.write("\n\n")

        for k, v in file_dic.items():
            fid.write("{} = {}".format(k, v))
            fid.write("\n")


def make_geometry(opts: PreProcOpts, file_name: Optional[str] = None, image_name: str = 'corrected',
                  xsize: Optional[int] = None, ysize: Optional[int] = None,
                  mask: bool = True, write_mask: bool = False, **kwargs):
    """Generates a CrystFEL geometry file from a PreProcOpts object

    Args:
        opts (PreProcOpts): options object holding the required information, which are
            the ellipticity parameters and camera length, pixel size, wave length, and 
            image dimensions
        file_name (str, optional): filename of a geometry file to be written. If None,
            returs the file contents as a dict instead. Defaults to None.
        image_name (str, optional): label of the diffraction data stack in the HDF5 files. 
            Defaults to 'corrected'.
        xsize (int, optional): x image size. If None, use that in opts. Defaults to None.
        ysize (int, optional): y image size. If None, use that in opts. Defaults to None.
        mask (bool, optional): Include reference to a pixel mask. Defaults to True.
        write_mask (bool, optional): Create a file `pxmask.h5` containing the pixel mask
            as defined in the options object (required as CrystFEL needs the masks as HDF5).
            Defaults to False.
        **kwargs: further lines to be included into the geometry file (or overwritten)

    Returns:
        Optional[dict]: if file_name=None, a dict with the geometry file rows is returned
    """
    
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
    
    if mask:
        par.update({'mask': '/mask',
            'mask_file': 'pxmask.h5',
            'mask_good': '0x01',
            'mask_bad': '0x00'})
        
    if write_mask:
        with h5py.File('pxmask.h5', 'w') as fh:
            fh['/mask'] = 1-imread(opts.pxmask)

    par.update(kwargs)
    
    if file_name is not None:
        dict2file(file_name, par, 
                  header=';Lambda detector file generated by diffractem.\n'
                  f';Ellipticity correction with ratio {opts.ellipse_ratio}, angle {opts.ellipse_angle} deg.')
        return None
    
    else:
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
    """Analyze a `hkl`-file triplet as generated by `partialator` (comprising `hkl`, `hkl1, `hkl2`).
    Uses the `check_hkl` and `compare_hkl` tools included with *CrystFEL*, and mangles their output
    into something a bit more friendly.

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


def viewing_widget(ds_disp, shot=0, Imax=30, log=False):
    """Interactive viewing widget for use in Jupyter notbeooks.

    Args:
        ds_disp ([type]): [description]
        Imax (int, optional): [description]. Defaults to 30.
        log (bool, optional): [description]. Defaults to False.
    """
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt

    output = widgets.Output()
    with output:
        fh, ax = plt.subplots(1,1, constrained_layout=True)
        
    have_peaks = 'nPeaks' in ds_disp.stacks
    have_center = 'center_x' in ds_disp.shots
    
    img_stack = ds_disp.diff_data
    
    if max(ds_disp.diff_data.chunks[0]) > 10:
        warn(f'Diffraction data chunks are large (up to {max(ds_disp.diff_data.chunks[0])} shots). If their '
             'computation is heavy or your disk is slow, consider rechunking the dataset in a smart way for display.')
    
    fh.canvas.toolbar_position='bottom'    
    fh.canvas.header_visible=False    
    ih = ax.imshow(img_stack[shot,...].compute(scheduler='threading'), vmin=0, vmax=Imax, cmap='gray_r')
    if have_peaks:
        sc = ax.scatter([], [], c='g', alpha=0.1)
    if have_center:
        cx, cy = (plt.axvline(ds_disp.shots.loc[0,'center_x'], c='b', alpha=0.2), 
                  plt.axhline(ds_disp.shots.loc[0,'center_y'], c='b', alpha=0.2))
    ax.axis('off')
    
    # symmetrize figure

    w_shot = widgets.IntSlider(min=0, max=img_stack.shape[0], step=1, value=shot)
    w_selected = widgets.ToggleButton(False, description='selected')
    w_indicator = widgets.Label(f'{ds_disp.shots.selected.sum()} of {len(ds_disp.shots)} shots selected.')
    w_info = widgets.Textarea(layout=widgets.Layout(height='100%'))
    w_vmax = widgets.FloatText(Imax, description='Imax')
    w_log = widgets.Checkbox(log, description='log')
    # w_info_parent = widgets.Accordion(children=[w_info])
    
    def update(shot=shot, vmax=Imax, log=log):
        shdat = ds_disp.shots.loc[shot]
        w_selected.value = bool(shdat.selected)
        w_info.value = '\n'.join([f'{k}: {v}' for k, v in shdat.items()])
        if log:
            ih.set_data(np.log10(img_stack[shot,...].compute(scheduler='single-threaded')))
            ih.set_clim(0.1, np.log10(vmax))            
        else:           
            ih.set_data(img_stack[shot,...].compute(scheduler='single-threaded'))
            ih.set_clim(0, vmax)
        if have_peaks:
            sc.set_offsets(np.stack((ds_disp.peakXPosRaw[shot,:ds_disp.shots.loc[shot,'num_peaks']].compute(scheduler='single-threaded'), 
                    ds_disp.peakYPosRaw[shot,:ds_disp.shots.loc[shot,'num_peaks']].compute(scheduler='single-threaded'))).T)    
        if have_center:
            cx.set_xdata(ds_disp.shots.loc[shot,'center_x'])
            cy.set_ydata(ds_disp.shots.loc[shot,'center_y'])
            
        # ax.set_title(f'{shdat.file}: {shdat.Event}\n {shdat.num_peaks} peaks')
        fh.canvas.draw()
        
    def set_selected(val):
        ds_disp.shots.loc[w_shot.value, 'selected'] = val['new']
        w_indicator.value =  f'{ds_disp.shots.selected.sum()} of {len(ds_disp.shots)} shots selected.'
    
    update()
    
    interactive(update, shot=w_shot, vmax=w_vmax, log=w_log)
    w_selected.observe(set_selected, 'value')

    ui = widgets.VBox([widgets.HBox([widgets.VBox([w_info, 
                                     w_shot]), 
                                     output]), 
                       widgets.HBox([w_selected, 
                                     w_indicator, 
                                     w_vmax, 
                                     w_log])]
                      )

    display(ui)