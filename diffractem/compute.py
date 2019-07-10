import numpy as np
import dask.array as da


def map_reduction_func(imgs, fun, *args, output_len=1, dtype=np.float, **kwargs):
    """
    Use dask array map blocks for functions that return a numpy vector of values (e.g. fit functions or 1D profiles)
    :param imgs: image stack as dask array, stacked along dimension 0
    :param fun: function to apply, needs to be able to process image stacks
    :param args: positional arguments to be supplied to the function. Note that these have to have three dimensions
    :param output_len: length of output numpy vector
    :param dtype: data type of output numpy vector
    :param kwargs: keyword arguments to be supplied to the function
    :return:
    """

    assert isinstance(imgs, da.core.Array)

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
    # print(fun)
    # print([type(a) for a in args_new])
    # print({kw: type(v) for kw, v in kwargs.items()})
    out = imgs.map_blocks(fun, *args_new, chunks=(imgs.chunks[0], output_len),
                          drop_axis=(1, 2), new_axis=1, dtype=dtype, **kwargs)
    return out


