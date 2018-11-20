import os.path
from glob import glob

import h5py
import numpy as np
import tifffile
from dask import array as da, delayed

from lambda_tools.io import load_meta, save_lambda_img, load_lambda_img
from lambda_tools.proc2d import correct_dead_pixels, apply_flatfield


def process_files(files='*.nxs', output_dir='.', postfix='_proc', output_shape='asinput',
                  stack_size=100, chunks=None, ops=(correct_dead_pixels, apply_flatfield),
                  formats=('nxs',), make_average=False, max_workers=None,
                  pxmask_tif=None, reference_tif=None, dtype=None, **kwargs):
    """
    Process a single, or a whole set of Lambda nxs stack files, using a defined series of operations. For the output,
    it can re-distribute the stacks, eg. to unpack them into single images, or join sub-stacks into one. As input,
    it requires nxs files, preferably with metadata included.
    For the output, there are two modes: (1) the function writes nxs files only, which keep the initial file
    properties. The output files can be larger than memory. (2) the function writes any combination of formats (as
    save_lambda_img), but the output files must fit into memory.
    :param files: Input nxs files, either as a list/tuple, or as a wildcard pattern for glob.
    :param output_dir: Destination directory. Better not the same as the input.
    :param output_shape:
    :param stack_size: Stack size of the output file, if output_shape is explicit
    :param maxchunk: Number of chunks to be processed at once
    :param ops: Operations to be performed on the files
    :param formats: Output formats. If nxs only, the function can handle larger-than-memory datasets.
    :param make_average: Make averages of output (XXX change to sum?)
    :param pxmask_tif:
    :param reference_tif:
    :param kwargs:
    :return:
    """

    assert(output_shape in ['asinput', 'single', 'merge', 'explicit'])

    nxs_only = (len(formats) == 1) and (formats[0].lower() == 'nxs')

    if isinstance(files, str):
        imgfiles = glob(files)
        imgfiles.sort()
    elif isinstance(files, list) or isinstance(files, tuple):
        imgfiles = files
    else:
        raise ValueError('files must be a file pattern or a list of filenames!')

    # get reference data
    if not reference_tif is None:
        with tifffile.TiffFile(reference_tif) as tif:
            reference = tif.asarray()
    else:
        reference = None

    # get dead pixel mask
    if not pxmask_tif is None:
        with tifffile.TiffFile(pxmask_tif) as tif:
            pxmask = tif.asarray()
    else:
        pxmask = None

    # get metadata
    meta = load_meta(imgfiles)

    # get dataset handle(s) and dask array(s) from NXS file(s).
    datasets = [h5py.File(fn, 'r')['entry/instrument/detector/data'] for fn in imgfiles]
    img_array = da.concatenate([da.from_array(ds, ds.chunks) for ds in datasets])

    if chunks is not None:
        img_array = img_array.rechunk((chunks, -1, -1))

    img_fn = [output_dir + '/' + os.path.basename(fn).rsplit('.',1)[0] + postfix for fn in imgfiles]

    if dtype is None:
        theDtype = img_array.dtype
    else:
        theDtype = dtype

    img_out = img_array.map_blocks(process_stack, dtype=theDtype, ops=ops, execution='stack',
                                   pxmask=pxmask, reference=reference, **kwargs)

    # make slice ranges for output files
    if output_shape == 'asinput':
        shots = [ds.shape[0] for ds in datasets]
        fns_out = img_fn
        meta_out = meta

    elif output_shape == 'merge':
        shots = [img_array.shape[0], ]
        fns_out = [img_fn[0] + '_merged',]
        meta_out = [meta[0], ]

    elif output_shape == 'single':
        shots = [1,] * img_array.shape[0]
        fns_out = []
        meta_out = []
        for fn, md in zip(img_fn, meta):
            fns_out.extend([fn + '_{:05d}'.format(ii) for ii in range(shots)])
            meta_out.extend([md for i in range(shots)])

    elif output_shape == 'explicit':
        shots = []
        fns_out = []
        meta_out = []
        for ds, fn, md in zip(datasets, img_fn, meta):
            dm = divmod(ds.shape[0], stack_size)
            shots.extend([stack_size,] * dm[0])
            shots.append(dm[1])
            fns_out.extend([fn + '_{:05d}'.format(ii) for ii in range(0, ds.shape[0], stack_size)])
            meta_out.extend([md for i in range(0, ds.shape[0], stack_size)])

    else:
         raise ValueError('Output shape not recognized')

    return img_out, fns_out, meta_out, shots

    # define the function to generate the output as a delayed function
    @delayed
    def writefile(fn, md, dat):
        save_lambda_img(dat, fn, formats=formats, meta=md,
                        make_average=make_average, compression=datasets[0].compression_opts)
        return 0

    done = []
    curr_shot = 0

    for sh, fn, md in zip(shots, fns_out, meta_out):
        io = img_out[curr_shot:curr_shot+sh,...]
        done.append(writefile(fn, md, io))
        curr_shot += sh

    # finally: do it! TODO: MODE FOR LARGER THAN MEMORY FILES
    # return compute(done)

    return img_out, fns_out, meta_out, shots


def map_reduction_func(darr, fun, *args, output_len=1, dtype=np.float, **kwargs):
    """
    Use dask array map blocks for functions that return a numpy vector of values (e.g. fit functions or 1D profiles)
    :param darr: image stack as dask array, stacked along dimension 0
    :param fun: function to apply, needs to be able to process image stacks
    :param args: positional arguments to be supplied to the function. Note that these have to have three dimensions
    :param output_len: length of output numpy vector
    :param dtype: data type of output numpy vector
    :param kwargs: keyword arguments to be supplied to the function
    :return:
    """

    assert isinstance(darr, da.core.Array)

    args_new = []
    for arg in args:
        # broadcasting on arrays works on the last dimension, whereas the stack is in the first. This may cause trouble
        # if a parameter array is 1D or 2D
        if isinstance(arg, da.core.Array) or isinstance(arg, np.ndarray):
            if arg.ndim == 1:
                #print('upcasting 1D')
                arg = arg[:, np.newaxis, np.newaxis]
            elif arg.ndim == 2:
                #print('upcasting 2D')
                arg = arg[:, :, np.newaxis]
        args_new.append(arg)

    out = darr.map_blocks(fun, *args_new, chunks=(darr.chunks[0], output_len),
                          drop_axis=(1,2), new_axis=1, dtype=dtype, **kwargs)
    return out


def process_stack(imgs, ops, execution='threads', **kwargs):
    """
    Applies correction function, or a pipeline thereof, to a stack of images, using different ways of execution.
    Essentially very similar to map function of hyperspy.
    :param imgs: 3D image stack or 2D single image
    :param ops: function handle to processing function, or list/tuple thereof.
    :param execution: method of execution. Options are
        'threads': process each image in parallel using threads
        'processes': process each image in parallel using processes
        'stack': pass the entire stack to the ops functions. All ops must be able to handle stack inputs
        'loop': loop through the images. Use for debugging only!
    :param kwargs: keyword arguments to be passed to the correction functions
    :return: the processed stack/image
    """

    assert execution.lower() in ('threads', 'processes', 'stack', 'loop')

    is_2D = len(imgs.shape) < 3

    if is_2D:
        imgs.shape = (1,imgs.shape[0],imgs.shape[1])

    imgs_out = imgs.copy()

    if not (isinstance(ops, list) or isinstance(ops, tuple)):
        ops = (ops,)

    if execution.lower() in ['threads', 'processes']:
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        if execution.lower=='processes':
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor
        for func in ops:
            imgs_out = Executor().map(lambda img: func(img, **kwargs), imgs_out)
            imgs_out = np.stack(imgs_out)

    elif execution.lower=='loop':
        for ii, img in enumerate(imgs_out):
            for func in ops:
                imgs_out[ii,:,:] = func(img, **kwargs)

    else:
        for func in ops:
            imgs_out = func(imgs_out, **kwargs)

    if is_2D:
        imgs_out.shape = (imgs_out.shape[1],imgs_out.shape[2])

    return imgs_out


def load_process_save(file_or_ds, ops=(correct_dead_pixels, apply_flatfield),
                      range=None, base_fname='images', formats=('mrc',),
                      make_average=False, truncate_invalid=50, meta=None, **kwargs):
    """DEPRECATED!!! All-in-one function for processing a single image (stack).
    Keyword arguments are propagated to the functions contained in the ops list. """

    print('Processing {}...'.format(file_or_ds))

    imgs = load_lambda_img(file_or_ds, range=range)

    if ops is None:
        imgs_corr = imgs
    else:
        imgs_corr = process_stack(imgs, ops, **kwargs)

    if truncate_invalid:
        imgs_corr = imgs_corr[:,:,truncate_invalid:-truncate_invalid]

    #if make_average:
    #    fn = base_fname + '_stack'
    #else:
    fn = base_fname

    save_lambda_img(imgs_corr, fn, formats, make_average=make_average, meta=meta, **kwargs)