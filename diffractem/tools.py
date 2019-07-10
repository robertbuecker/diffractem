# Miscellaneous supporting tool functions

from astropy.convolution import Gaussian2DKernel, convolve

from . import gap_pixels
from .io import *
from .proc2d import correct_dead_pixels


def strip_file_path(df: pd.DataFrame, add_folder=False):
    splt = df.file.str.rsplit('/', 1, True)
    if add_folder:
        if splt.shape[1] == 2:
            return df.assign(file=splt.iloc[:,-1], folder=splt.iloc[:,0])
        else:
            return df.assign(file=splt.iloc[:,-1], folder='.')
    else:
        return df.assign(file=splt.iloc[:,-1])


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


def call_indexamajig(input, geometry, output='im_out.stream', cell=None, im_params=None, index_params=None,
                     procs=40, exc='indexamajig'):

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
           'mask': '/%/data/pxmask_centered_fr',
           'mask_good': '0x01',
           'mask_bad': '0x00',
           'data': '/%/data/centered_fr',
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

