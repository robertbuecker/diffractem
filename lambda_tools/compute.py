import numpy as np
from dask import array as da

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
