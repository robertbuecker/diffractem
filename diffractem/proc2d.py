import tifffile
from tifffile import imread
import numpy as np
import dask.array as da
import dask
from dask.distributed import Client
from numba import jit, prange, int64
from . import gap_pixels, nexus
from .pre_proc_opts import PreProcOpts
from scipy import optimize, sparse, special, interpolate
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import disk
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage.filters import median_filter
from functools import wraps
from typing import Optional, Tuple, Union, List, Callable, Dict
from warnings import warn, catch_warnings, simplefilter
import pandas as pd
import h5py


def stack_nested(data_list: Union[tuple, list, dict], func: Callable = np.stack): 
    """Applies a numpy/dask concatenation/stacking function recursively to a recursive 
    python collection (tuple/list/dict) containing numpy or dask arrays on the lowest level.

    Args:
        data_list (Union[tuple, list, dict]): Collection of numpy arrays (can be recursive)
        func (Callable, optional): Concatenation function to apply. Defaults to np.stack.

    Returns:
        same as data_list: tuple/list/dict with concatenated/stacked numpy arrays
    """
    
    
    if np.ndim(data_list) == 0:
        data_list = [data_list]
    if isinstance(data_list[0], tuple):
        return tuple(stack_nested(arr, func) for arr in zip(*data_list))
    elif isinstance(data_list[0], list):
        return list(stack_nested(arr, func) for arr in zip(*data_list))
    elif isinstance(data_list[0], dict):
        return {k: stack_nested(list(o[k] for o in data_list), func) for k in data_list[0].keys()}
    else:
        return func(data_list)


def loop_over_stack(fun):
    """
    Decorator to (sequentially) loop a 2D processing function over a stack. 
    
    In brief, if you have a function that either modifies a (single) image or extracts some reduced
    data from it, this decorator wraps it such that it can operate on a whole stack of images.
    
    Works on all functions with signature fun(imgs: np.ndarray, *args, **kwargs),
    where imgs is a numpy 3D stack or a 2D single image. It has to return either
    a numpy array,n which case it returns a stacked array of the function output, 
    or a collection containing numpy arrays, each of which is stacked individually.
    If any of the positional/named arguments is an iterable of the same length 
    as the image stack, it is distributed over the function calls for each 
    image.
    
    Note:
        `loop_over_stack` only works on functions eating numpy arrays, *not* dask arrays.
        If you want to apply a function to a dask-array image stack, you have to *additionally*
        wrap it in `dask.array.map_blocks`, `diffractem.dataset._map_sub_blocks`, 
        `diffractem.compute.map_reduction_func` or similar.
         
    Args:
        fun (Callable): function to be decorated
    
    Returns:
        Callable: function that loops over an image stack automatically

    """
    #TODO: handle functions with multiple outputs, and return a list of ndarrays
    #TODO: allow switches for paralellization

    @wraps(fun)
    def loop_fun(imgs, *args, **kwargs):
        # this is NOT intended for dask arrays!
        if not isinstance(imgs, np.ndarray):
            raise TypeError(f'loop_over_stack only works on numpy arrays (not dask etc.). '
                            f'Passed type to {fun} is {type(imgs)}')

        if imgs.ndim == 2:
            return fun(imgs, *args, **kwargs)

        elif imgs.shape[0] == 1:
            # some gymnastics if arrays are on weird dimensions (often after map_blocks)
            args = [a.squeeze() if isinstance(a, np.ndarray) else a for a in args]
            kwargs = {k: a.squeeze() if isinstance(a, np.ndarray) else a for k, a in kwargs.items()}
            return stack_nested([fun(imgs.squeeze(), *args, **kwargs)])

        #print('Applying {} to {} images of shape {}'.format(fun, imgs.shape[0], imgs.shape[1:]))

        def _isiterable(arg):
            try:
                (a for a in arg)
                return True
            except TypeError:
                return False

        iter_args = []
        for a in args:
            if _isiterable(a) and len(a) == len(imgs):
                if isinstance(a, np.ndarray):
                    a = a.squeeze()
                iter_args.append(a)
            else:
                # iter_args.append(repeat(a, lenb(imgs)))
                iter_args.append([a]*len(imgs))

        iter_kwargs = []
        for k, a in kwargs.items():
            if _isiterable(a) and len(a) == len(imgs):
                if isinstance(a, np.ndarray):
                    a = a.squeeze()
                iter_kwargs.append(a)
            else:
                # iter_kwargs.append(repeat(a, len(imgs)))
                iter_kwargs.append([a] * len(imgs))

        out = []

        for arg in zip(imgs, *(iter_args + iter_kwargs)):
            theArgs = arg[1:1+len(args)]
            theKwargs = {k: v for k, v in zip(kwargs, arg[1+len(args):])}
            # print('Arguments: ', theArgs)
            # print('KW Args:   ', theKwargs)
            out.append(fun(arg[0], *theArgs, **theKwargs))

        if not out:
            # required for dask map_blocks init runs
            # print('Looping of',fun,'requested for zero-size input.')
            return np.ndarray(imgs.shape, dtype=imgs.dtype)

        try:
            # print(type(out), type(out[0]))
            return stack_nested(out)
                
        except ValueError as err:
            print('Function',fun,'failed for output array construction.')
            raise err

    return loop_fun


@loop_over_stack
def _generate_pattern_info(img: np.ndarray, opts: PreProcOpts, 
                           reference: Optional[np.ndarray] = None, 
                           pxmask: Optional[np.ndarray] = None,
                           centers: Optional[np.ndarray] = None,
                           lorentz_fit: Optional[bool] = True) -> dict:
    """
    'Macro' function computing information from diffraction data and returning it
    as a dictionary. Primarily intended to be called from `get_pattern_info`.
    
    Note:
        This function is different from most in `proc2d` in that it returns a
        dictionary, *no* a `np.ndarray`. This has, among others, the implication, that
        it cannot be called through the dask array interface via `map_blocks`.

    Args:
        img (np.ndarray): diffraction image or stack thereof as numpy array.
        opts (PreProcOpts): pre-processing options.
        reference (Optional[np.ndarray], optional): reference image for flat-field.
            correction. If None, grabs the file name from the options file and loads it. 
            This is discouraged as it requires reloading it over and over. Defaults to None.
        pxmask (Optional[np.ndarray], optional): similar, for pixel mask. Defaults to None.

    Returns:
        dict: Diffraction pattern information.
    """
    #TODO consider using a NamedTuple for return values instead of a dict
    
    # computations on diffraction patterns. To be called from get_pattern_info.
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
        
    from diffractem.proc_peaks import _ctr_from_pks
    
    # apply flatfield and dead-pixel correction to get more accurate COM
    # CONSIDER DOING THIS OUTSIDE GET PATTERN INFO!
    img = apply_flatfield(img, reference, keep_type=False)
    img = correct_dead_pixels(img, pxmask, strategy='replace', mask_gaps=False, replace_val=-1)
    
    if centers is None:
        # thresholded center-of-mass calculation over x-axis sub-range
        img_ct = img[:,(img.shape[1]-opts.com_xrng)//2:(img.shape[1]+opts.com_xrng)//2]
        com = center_of_mass(img_ct, threshold=opts.com_threshold*np.quantile(img_ct,1-5e-5)) + [(img.shape[1]-opts.com_xrng)//2, 0]
        
        if lorentz_fit:
        # Lorentz fit of direct beam
            lorentz = lorentz_fast(img, com[0], com[1], radius=opts.lorentz_radius,
                                                limit=opts.lorentz_maxshift, scale=7, threads=False)
            
            x0, y0 = lorentz[1], lorentz[2]
        else:
            x0, y0 = com[0], com[1]
            lorentz = [np.nan] * 4
             
    else:
        # print(centers.shape)
        x0, y0 = centers[0], centers[1]
        # print(centers)
        lorentz = [np.nan] * 4
        com = [np.nan] * 2
    
    # Get peaks using peakfinder8. Note that pf8 parameters are taken straight from the options file,
    # with automatic underscore/hyphen replacement.
    # Note that peak positions are CXI convention, i.e. refer to pixel _center_
    if opts.find_peaks:
        peak_data = get_peaks(img, x0, y0, pxmask=pxmask, max_peaks=opts.max_peaks,
                                **{k.replace('-','_'): v for k, v in opts.peak_search_params.items()},
                                as_dict=True)
        # print(peak_data.keys())
    else:
        peak_data = {'nPeaks': 0, 'peakXPosRaw': np.zeros((opts.max_peaks,)), 
                     'peakYPosRaw': np.zeros((opts.max_peaks,)),
                     'peakTotalIntensity': np.zeros((opts.max_peaks,))}
    
    if opts.friedel_refine and (peak_data['nPeaks'] >= opts.min_peaks):  
        
        # prepare peak list. Note the .5, as _ctr_from_pks expects CrystFEL peak convention,
        # i.e. positions refer to pixel _corner_
        pkl = np.stack((peak_data['peakXPosRaw'] + .5, peak_data['peakYPosRaw'] + .5, 
                        peak_data['peakTotalIntensity']), -1)[:int(peak_data['nPeaks']),:]
        if opts.friedel_max_radius is not None:
            rsq = (pkl[:, 0] - x0) ** 2 + (pkl[:, 1] - y0) ** 2
            pkl = pkl[rsq < opts.friedel_max_radius ** 2, :]
        
        ctr_refined, cost, _ = _ctr_from_pks(pkl, np.array([x0, y0]), int_weight=False, 
                    sigma=opts.peak_sigma)
        
    else:
        ctr_refined, cost = np.array([x0, y0]), np.nan
    
    # print(ctr_refined, x0, y0)
    
    # virtual ADF detectors
    adf1 = apply_virtual_detector(img, opts.r_adf1[0], opts.r_adf1[1], ctr_refined[0], ctr_refined[1])
    adf2 = apply_virtual_detector(img, opts.r_adf2[0], opts.r_adf2[1], ctr_refined[0], ctr_refined[1])

    pattern_info = {'com_x': com[0], 'com_y': com[1],
                    'lor_pk': lorentz[0], 
                    'lor_x': lorentz[1],
                    'lor_y': lorentz[2],
                    'lor_hwhm': lorentz[3],
                    'center_x': ctr_refined[0],
                    'center_y': ctr_refined[1],
                    'center_refine_score': cost,
                    'adf1': adf1,
                    'adf2': adf2,
                    'shift_x_mm': -1e3 * opts.pixel_size * ctr_refined[0],
                    'shift_y_mm': -1e3 * opts.pixel_size * ctr_refined[1],
                    'num_peaks': peak_data['nPeaks'],
                    'peak_data': peak_data}
        
    return pattern_info


def get_pattern_info(img: Union[np.ndarray, da.Array], opts: PreProcOpts, client: Optional[Client] = None, 
                     reference: Optional[np.ndarray] = None, 
                     pxmask: Optional[np.ndarray] = None, 
                     centers: Optional[Union[np.ndarray, da.Array]] = None,
                     lorentz_fit: Optional[bool] = True,
                     lazy: bool = False, sync: bool = True,
                     errors: str = 'raise', via_array: bool = False,
                     output_file: Optional[str] = None, 
                     shots: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, dict]:
    """'Macro' function for getting information about diffraction patterns.
    
    `get_pattern_info` finds diffraction peaks and computes information such as pattern center on a given diffraction 
    pattern or stack thereof. By default (`lazy=False` and `sync=True`) it will return a pandas DataFrame containing
    general information on each pattern, and a dict holding the found peaks in CXI format.
    
    The options for preprocessing are passed as a `PreProcOpts` object.
    
    Note:
        This function is essentially a smart wrapper around `prof2d._generate_pattern_info`. If you'd like to change
        what is actually calculated and how, that is the function to modify!
    
    Note:
        As this function is computationally heavy, it is **very** advisable to use a *dask.distributed* cluster for 
        computation, with a client object supplied to the function call.

    Args:
        img (Union[np.ndarray, da.Array]): stack of diffraction patterns, typically a dask array
        opts (PreProcOpts): pre-processing options.
        client (Optional[Client], optional): Client object for dask.distributed cluster. If None, runs
            computation by simply calling `compute` on the stack dask array (discouraged). Defaults to None.
        reference (Optional[np.ndarray], optional): Flat-field reference image. If None, load the one specified in
            preprocessing options. Defaults to None.
        pxmask (Optional[np.ndarray], optional): Pixel mask image. If None, load the one specified in
            preprocessing options. Defaults to None.
        centers (np.array or da.Array, optional): N x 2 matrix with known centers of all diffraction patterns. If set,
            the center-of-mass and Lorentz fit steps are skipped. Depending on the setting of `opts.friedel_refine`,
            Friedel-mate center refinement is still performed. Defaults to None.
        lazy (bool, optional): Return `dask.delayed` objects for pattern info generation tasks instead of the final
            results. Mostly useful for debugging or embedding into more complex workflows. Defaults to False.
        sync (bool, optional): Immediately compute pattern info. If False, returns futures to pattern info dictionaries
            instead of DataFrame and peak dict. Defaults to True.
        errors (str, optional): Behavior if errors arise during eager computation (i.e., `lazy=False`, `sync=True`). If
            'raise', errors are raised, if 'skip', they are skipped, and the final data is missing the corresponding
            shots, which needs to be handled downstream to avoid making a mess. Defaults to 'raise'.
        via_array (bool, optional): Modify calculation such that it avoids `dask.delayed` objects.
            This drastically improves the scheduling behavior for large datasets. It is also required if you
            supply the pattern centers to the function. However, precludes the use of lazy and sync. Defaults to False.
        output_file (str, optional): Filename to store calculation results into. The file will be a valid diffractem-type
            data file that can be loaded using Dataset objects.
        shots (pd.DataFrame, optional): Dataframe of shot data of same height as the image array. If not None,
            its columns will be joined to those of the shot data for storing the results into the output file.

    Returns:
        Tuple[pd.DataFrame, dict]: pandas DataFrame holding general pattern information, and dict holding CXI-format
            peaks. (note that return values are different when using `lazy=True` or `sync=False` - see above)
    """
    #TODO could this be refactored into dataset, automatically applying it to the diffraction data set?
    #TODO would including an option to return cts on top of res_del make sense?
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
#     print(type(pxmask))

    if len(img.shape) == 2:
        img = img[np.newaxis, ...]
    
    if isinstance(img, da.Array) and not via_array:
        if centers is not None:
            raise ValueError('If pattern centers are given, you have to set via_array=True.')
        cts = img.to_delayed().squeeze()
        res_del = [dask.delayed(_generate_pattern_info, nout=1, 
                                pure=True)(c, opts=opts, reference=reference, pxmask=pxmask, lorentz_fit=lorentz_fit,
                                           dask_key_name=f'pattern_info-{ii}') for ii, c in enumerate(cts)]
        if lazy:
            return res_del
        if client is not None:
            ftrs = client.compute(res_del)
            print(f'Running get_pattern_info on cluster at {client.scheduler_info()["address"]}. \n'
                  f'Watch progress at {client.dashboard_link} (or forward port if remote).')
            if not sync:
                return ftrs
            alldat = stack_nested(client.gather(ftrs, errors=errors), func=np.concatenate)
        else:
            warn('get_pattern_info is run on a dask array without distributed client - might be slow!')
            alldat = dask.compute()
            
    elif isinstance(img, da.Array) and via_array:
        # do extra step by casting output of _generate_pattern_info into a dask array and 
        # keep using the dask array API instead of the delayed api as above. For yet not understood
        # reasons this leads to a much better behavior of the dask scheduler and yields identical results.
        # it's just horribly inelegant.
        
        # function to turn output of _generate_pattern_info into a dask array
        _encode_info = lambda info: np.concatenate([np.stack([v for k, v in sorted(info.items()) if k != 'peak_data'], axis=1),
                np.hstack([v.reshape(v.shape[0],-1) for k, v in sorted(info['peak_data'].items())])], axis=1)

        # compute info for a single image to get structure of output
        template = _generate_pattern_info(img[:1,...].compute(), opts, centers=None if centers is None else centers[:1,:])
        
        if centers is not None:
            centers = da.from_array(centers, chunks=(img.chunks[0], 2)).reshape((-1, 2, 1))

        info_array = img.map_blocks(lambda img, centers: _encode_info(
            _generate_pattern_info(img, opts, centers=centers, lorentz_fit=lorentz_fit)), centers,
                                    dtype=np.float, drop_axis=[1,2], new_axis=1, 
                                    chunks=(img.chunks[0], _encode_info(template).shape[1]),
                                    name='pattern_info')
        # return info_array # for debugging purposes
        alldat = client.compute(info_array, sync=True)

        # recreate shot data table
        cols = [k for k in sorted(template) if k != 'peak_data']
        types = {k: v.dtype for k, v in sorted(template.items()) if k != 'peak_data'}
        shotdata = pd.DataFrame(alldat[:,:len(cols)], columns=cols).astype(types)
        
        # recreate peak info
        pk_cols = [(k, v.reshape(v.shape[0],-1).shape[1], v.dtype) 
                for k, v in sorted(template['peak_data'].items())]
        
        peakinfo, ii_col = {}, len(cols)
        for col, width, dt in pk_cols:
            peakinfo[col] = alldat[:, ii_col:ii_col+width].squeeze().astype(dt)
            ii_col += width
            
    elif isinstance(img, np.ndarray):
        alldat = _generate_pattern_info(img, opts=opts, reference=reference, 
                                        pxmask=pxmask, centers=centers, lorentz_fit=lorentz_fit)
        
        shotdata = pd.DataFrame({k: v for k, v in alldat.items() if isinstance(v, np.ndarray) and (v.ndim == 1)})
        peakinfo = alldat['peak_data']
        
    else:
        raise ValueError('Input image(s) must be a dask or numpy array.')
        
    if output_file is not None:
        with h5py.File(output_file, 'w') as fh:
            for k, v in peakinfo.items():
                fh.create_dataset('/entry/data/' + k, data=v, compression='gzip', chunks=(1,) + v.shape[1:])
            fh['/entry/data'].attrs['recommended_zchunks'] = -1
        shotdata_id = pd.concat([shots, shotdata], axis=1)
        nexus.store_table(shotdata_id, file=output_file, subset='entry', path='/%/shots')
        print('Wrote analysis results to file', output_file)
    
    return shotdata, peakinfo    


@loop_over_stack
def _get_corr_img(img: np.ndarray, 
                      x0: np.ndarray, y0: Union[np.ndarray, None], 
                      nPeaks: Union[np.ndarray, None], 
                      peakXPosRaw: Union[np.ndarray, None], 
                      peakYPosRaw: Union[np.ndarray, None], 
                      opts: PreProcOpts,
                      reference: Optional[Union[np.ndarray, str]] = None, 
                      pxmask: Optional[Union[np.ndarray, str]] = None):
    """Inner function for full correction pipeline. To be called from `correct_image`. Other than
    that function, this one can only run on numpy arrays.
    
    Please see doumentation of `correct_image` for further documentation.
    """
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
    
    img = img.astype(np.float32)
    if opts.correct_saturation:
        img = apply_saturation_correction(img, opts.shutter_time, opts.dead_time, opts.dead_time_gap_factor)
    img = apply_flatfield(img, reference=reference)

    # here, _always_ choose strategy='replace'. Interpolation will only be done on the final step
    # img = correct_dead_pixels(img, pxmask=pxmask, strategy='replace', mask_gaps=opts.mask_gaps)

    img = remove_background(img, x0, y0, nPeaks, peakXPosRaw, peakYPosRaw, pxmask=None)
    return img
    # has to be re-done after background correction
    # TODO THIS IS A TRAIN WRECK. FIX ME
    img = correct_dead_pixels(img, pxmask, strategy='interpolate' if opts.interpolate_dead else 'replace', mask_gaps=opts.mask_gaps)
    
    return img


def correct_image(img: Union[np.ndarray, da.Array], opts: PreProcOpts, 
                  x0: Union[None, np.ndarray, da.Array, pd.Series] = None, 
                  y0: Union[None, np.ndarray, da.Array, pd.Series] = None, 
                  peakinfo: Union[None, Dict[str, Union[np.ndarray, da.Array]]] = None, 
                  reference: Union[None, Union[np.ndarray, str]] = None, 
                  pxmask: Union[None, Union[np.ndarray, str]] = None) -> Union[np.ndarray, da.Array]:
    """Runs correction pipeline on stack of diffraction images (numpy or dask). 
    
    The correction pipeline comprises flat-field, saturation and dead-pixel correction, as well as 
    background subtraction, optionally including exclusion of diffraction peaks for computation of the
    background (recommended).
        
    Note:
        This function essentially wraps `proc2d._get_corr_image` with smart features to take care
        of dask input arrays. If you want to change the 
        correction pipeline, that is the function to modify.

    Args:
        img (Union[np.ndarray, da.Array]): Diffraction pattern stack
        opts (PreProcOpts): Pre-processing options. Options used are: (...)
        x0 (Union[None, np.ndarray, da.Array, pd.Series], optional): Pattern X centers 
            (None: use image center). Defaults to None.
        y0 (Union[None, np.ndarray, da.Array, pd.Series], optional): Pattern Y centers 
            (None: use image center). Defaults to None.
        peakinfo (Union[None, Dict[str, Union[np.ndarray, da.Array]]], optional): Diffraction peak dict 
            in CXI format  (None: no peak exclusion during background subtraction). Defaults to None.
        reference (Union[None, Union[np.ndarray, str]], optional): Flat-field reference 
            (None: use reference file specified in options). Defaults to None.
        pxmask (Union[None, Union[np.ndarray, str]], optional): Pixel mask reference 
            (default: use reference file specified in options). Defaults to None.

    Returns:
        Union[np.ndarray, da.Array]: Corrected image stack of identical dimension as input stack.
    """
    
    if isinstance(img, np.ndarray):
        # take care of numpy image with dask arguments (just in case)
        innerargs = [x0, y0, peakinfo['nPeaks'], peakinfo['peakXPosRaw'], peakinfo['peakYPosRaw']]
        innerargs = [a.compute() if isinstance(a, da.Array) else a for a in innerargs]
        return _get_corr_img(img, *innerargs, opts, reference, pxmask)
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
    
    N = img.shape[0]
    
    if (x0 is None) or (y0 is None):
        x0 = y0 = None
        
    else:
        if not isinstance(x0, da.Array):
            x0 = da.from_array(x0.values if isinstance(x0, pd.Series) else x0, chunks=img.chunks[0])
            y0 = da.from_array(y0.values if isinstance(y0, pd.Series) else y0, chunks=img.chunks[0])
    
    if peakinfo is None:
        npk = pkx = pky = None
        
    else:     
        if not isinstance(peakinfo['nPeaks'], da.Array):
            peakinfo = {'nPeaks': da.from_array(peakinfo['nPeaks'], chunks=img.chunks[0]),
                    'peakXPosRaw': da.from_array(peakinfo['peakXPosRaw'], chunks=(img.chunks[0],-1)),
                    'peakYPosRaw': da.from_array(peakinfo['peakYPosRaw'], chunks=(img.chunks[0],-1))}

        npk = peakinfo['nPeaks'].reshape((N,1,1))
        pkx = peakinfo['peakXPosRaw'].reshape((N,1,-1))
        pky = peakinfo['peakYPosRaw'].reshape((N,1,-1))
    
    return da.map_blocks(_get_corr_img, img, x0.reshape((N,1,1)), y0.reshape((N,1,1)), 
                         npk, pkx, pky, 
                         reference=reference, pxmask=pxmask, opts=opts,
                         dtype=np.float32, chunks=img.chunks)


def analyze_and_correct(imgs: np.ndarray, opts: PreProcOpts, 
                        correct_non_hits: bool = False,
                        reference: Union[None, Union[np.ndarray, str]] = None, 
                        pxmask: Union[None, Union[np.ndarray, str]] = None) -> Tuple[np.ndarray, dict]:
    """Analyzes a diffraction pattern (centering and peak finding), and immediately applies a correction.
    
    This function combines `get_pattern_info` and `correct_image`, but works differently in that
    it does not inherently handle any lazy/parallel computations: it only simply loops over a numpy 
    array. It is hence especially useful to check if the preprocessing pipeline works on a small
    set, or to embed it into dask delayed objects for parallel execution *outside* the function, which 
    may be faster than `get_pattern_info` + `correct_image` (see example below).

    Args:
        imgs (np.ndarray): Input image stack as numpy array
        opts (PreProcOpts): pre-processing options
        correct_non_hits (bool, optional): Apply correction also to images that do not have 
            sufficient Bragg spots in them (as defined by opts.min_peaks). Defaults to False.
        reference (Union[None, Union[np.ndarray, str]], optional): Reference image as numpy
            array or TIF file name. If None, read file defined in options. Defaults to None.
        pxmask (Union[None, Union[np.ndarray, str]], optional): Pixel mask image as numpy
            array or TIF file name. If None, read file defined in options. Defaults to None.

    Returns:
        Tuple[np.ndarray, dict]: Corrected image stack and pattern info structure, as
            returned by `correct_image` and `get_pattern_info`, respectively.
        
        
    Example:
        To run a parallel computation efficiently, use this function like
        
        >>> results = [dask.delayed(proc2d.analyze_and_correct)(img_chunk, opts) \
                    for img_chunk in img_stack.to_delayed().ravel()]
        >>> dask.compute(results)
    """
    
    reference = imread(opts.reference) if reference is None else reference
    pxmask = imread(opts.pxmask) if pxmask is None else pxmask
    
    info = _generate_pattern_info(imgs, opts, reference=reference, pxmask=pxmask)
    hits = info['num_peaks'] >= opts.min_peaks

    if correct_non_hits or all(hits):
        imgs = _get_corr_img(imgs.astype(np.float32), info['center_x'], info['center_y'], 
                                        info['peak_data']['nPeaks'], 
                                        info['peak_data']['peakXPosRaw'], 
                                        info['peak_data']['peakYPosRaw'], 
                                        opts, reference=reference, pxmask=pxmask)
    else:
        imgs = imgs.astype(np.float32)
        imgs[hits,...] = _get_corr_img(imgs[hits,...], info['center_x'][hits,...], info['center_y'][hits,...], 
                                          info['peak_data']['nPeaks'][hits,...], 
                                          info['peak_data']['peakXPosRaw'][hits,...], 
                                          info['peak_data']['peakYPosRaw'][hits,...], 
                                          opts, reference=reference, pxmask=pxmask)
        
    return imgs, info


#TODO: this might not really belong here -> move to some tools module? Or make private?
def mean_clip(c: np.ndarray, sigma: float = 2.0) -> float:
    """Iteratively keeps only the values from the array that satisfies
        0 < c < c_mean + sigma*std 
        and return the mean of the array. Assumes the
        array contains positive entries, 
        if it does not or the array is empty returns -1 

    Args:
        c (np.ndarray): input value array
        sigma (float, optional): number of standard deviations away from the mean 
            that is used for mean calculation. Defaults to 2.0.

    Returns:
        float: Mean of clipped values
    """
    
    c = c[c>0]
    if not c.size:
        return -1
    delta = 1.0
    while delta:
        c_mean = np.mean(c)
        size = c.size
        c = c[c < c_mean + sigma*np.sqrt(c_mean)]
        delta = size-c.size
    return c_mean


#TODO: as before
def func_lorentz(p: Union[list, tuple, np.ndarray], 
                 x: Union[float, np.ndarray], 
                 y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Function that returns a Student't distribution or generalised Cauchy
    distribution in Two Dimensions(x,y):
    
    amp * [(1 + ((x-x_0)/scale)**2) + (1 + ((y-y_0)/scale)**2)] ** (-shape/2)

    Args:
        p (Union[list, tuple, np.ndarray]): Parameter array: [amp, x_0, y_0, scale, shape]
        x (Union[float, np.ndarray]): x coordinate(s)
        y (Union[float, np.ndarray]): y coordinate(s)

    Returns:
        Union[float, np.ndarray]: function value at (x, y)
    """
    return p[0]*((1+((x-p[1])/p[3])**2.0 + ((y-p[2])/p[3])**2.0)**(-p[4]/2.0))


@loop_over_stack
def lorentz_fit(img, amp: float = 1.0, 
                x_0: float = 0.0, y_0: float = 0.0, 
                scale: float = 5.0, shape: float = 2.0,
                threshold: float = 0):    
    """
    Fits a Lorentz profile to find the center (x_0, y_0) of a diffraction
    pattern, ignoring any pixels with values < threshold.
    
    The fit function is based on:

    amp * [(1 + ((x-x_0)/scale)**2) + (1 + ((y-y_0)/scale)**2)] ** (-shape/2)
    
    Build upon optimize.least_squares function  which is thread safe
    Note: least.sq is not. Analytical Jacobian has been added.
    
    Note:
        If possible (i.e. you leave shape at 2.0), do **not** use this function, 
        it's really slow. Instead use `lorentz_fast`.
    
    Args:
        see fit function above
        
    Returns:
        OptimizeResult: result of optimization
    
    """
    param = np.array([amp,x_0,y_0,scale, shape])
    def jac_lorentz(p, x, y, img):
        radius = ((x-p[1])/p[3])**2.0 + ((y-p[2])/p[3])**2.0
        func = ((1+radius)**(-p[4]/2.0-1.0))
        d_amp = ((1+radius)**(-p[4]/2.0)) 
        d_x0 = (x-p[1])/(p[3]**2.0)*p[0]*p[4]*func
        d_y0 = (y-p[2])/(p[3]**2.0)*p[0]*p[4]*func
        d_scale = p[0]*p[4]/p[3]*radius*func
        d_shape = -0.5*p[0]*((1+radius)**(-p[4]/2.0))*np.log(1+radius)
        return np.transpose([d_amp, d_x0,d_y0, d_scale, d_shape]/(img**0.5))   
    def func_error(p, x, y, img):
        return (func_lorentz(p,cut[1],cut[0])-img)/(img**0.5)
    cut = np.where(img > threshold)
    out = optimize.least_squares(func_error,param,jac_lorentz,loss='linear',
                                 max_nfev=1000,args=(cut[1],cut[0],img[cut]),
                                 bounds=([1.0,0.0,0.0,1.0, 1.0],np.inf))
    return out


@loop_over_stack
def lorentz_fast(img, x_0: float = None, y_0: float = None, amp: float = None, 
                 scale: float = 5.0, radius: float = None, limit: float = None,
                 threshold: int = 0, threads: bool = False, verbose: bool = False):
    """Fast Lorentzian fit for finding beam center; especially suited for refinement after a reasonable estimate
    (i.e. to a couple of pixels) has been made by another method such as truncated COM.
    Compared to the other fits, it always assumes a shape parameter 2 (i.e. standard Lorentzian with asymptotic x^-2).
    It can restrict the fit to only a small region around the initial value for the beam center, which massively speeds
    up the function. Also, it auto-estimates the intial parameters somewhat reasonably if nothing else is given.
    
    Args:
        img (float): input image or image stack. If a stack is supplied, it is serially looped. Not accepting dask directly.
        x_0 (float, optional): estimated x beam center. If None, is assumed to be in the center of the image. Defaults to None.
        y_0 (float, optional): analogous. Defaults to None.
        amp (float, optional): estimated peak amplitude. If None, is set to the 99.99% percentile of img. Defaults to None.
        scale (float, optional): peak HWHM estimate in pixels. Defaults to 5.0.
        radius (float, optional): radius of a box around x_0, y_0 where the fit is actually done. If None, the entire image is used. Defaults to None.
        limit (float, optional): If not None, the fit result is discarded if the found beam_center is further away than this value from
            the initial estimate. Defaults to None.
        threshold (int, optional): pixel value threshold below which pixels are ignored. Defaults to 0.
        threads (bool, optional): if True, uses scipy.optimize.least_squares, which for larger arrays (radius more than around 15)
            uses multithreaded function evaluation. Especially for radius < 50, this may be slower than single-threaded.
            In this case, best set to False. Defaults to False.
        verbose (bool, optional): if True, a message is printed on some occasions. Defaults to False.

    Returns:
        np.ndarray: numpy array of refined parameters [amp, x0, y0, scale]
    """
    if (x_0 is None) or (not np.isfinite(x_0)) or np.isnan(x_0):
        x_0 = img.shape[1] / 2
    if (y_0 is None) or (not np.isfinite(y_0)) or np.isnan(y_0):
        y_0 = img.shape[0] / 2
    if radius is not None:
        try:
            x1 = int(x_0 - radius)
            x2 = int(x_0 + radius)
            y1 = int(y_0 - radius)
            y2 = int(y_0 + radius)
        except ValueError as err:
            print('Weird:', x0, y0, radius)
            raise err
        if (x1 < 0) or (x2 > img.shape[1]) or (y1 < 0) or (y2 > img.shape[0]):
            print('Cannot cut image around peak. Centering.')
            x1 = int(img.shape[1] / 2 - radius)
            x2 = int(img.shape[1] / 2 + radius)
            y1 = int(img.shape[0] / 2 - radius)
            y2 = int(img.shape[0] / 2 + radius)
        img = img[y1:y2, x1:x2]
    else:
        x1 = 0
        y1 = 0
    if amp is None:
        try:
            amp = np.percentile(img, 99)
        except Exception as err:
            print('Something weird: {} Cannot get image percentile. Img size is {}. Skipping.'.format(err, img.shape))
            return np.array([-1, x_0, y_0, scale])

    cut = np.where(img > threshold)
    x = cut[1] + x1
    y = cut[0] + y1
    img = img[cut]
    norm = img ** 0.5

    function = lambda p: p[0] * ((1 + ((x - p[1]) / p[3]) ** 2.0 + ((y - p[2]) / p[3]) ** 2.0) ** (-1))
    error = lambda p: (function(p) - img) / norm

    # The Jacobian is not used anymore, but let's keep it here, just in case
    def jacobian(p):
        radius = ((x - p[1]) / p[3]) ** 2.0 + ((y - p[2]) / p[3]) ** 2.0
        func = ((1 + radius) ** (-2.0))
        d_amp = ((1 + radius) ** (-1.0))
        d_x0 = (x - p[1]) / (p[3] ** 2.0) * p[0] * 2.0 * func
        d_y0 = (y - p[2]) / (p[3] ** 2.0) * p[0] * 2.0 * func
        d_scale = p[0] * 2.0 / p[3] * radius * func
        res = np.stack((d_amp / norm, d_x0 / norm, d_y0 / norm, d_scale / norm), axis=-1)
        return res

    param = np.array([amp, x_0, y_0, scale])
    # print(param)

    try:
        if threads:
            # new algorithm: uses multithreaded evaluation sometimes, which is not always desired!
            # out = optimize.least_squares(error, param, jac=jacobian, loss='linear',
            #                             max_nfev=1000, method='lm', verbose=0,
            #                             x_scale=(amp, 1, 1, 5)).x
            out = optimize.least_squares(error, param, loss='linear',
                                        max_nfev=1000, method='lm', verbose=0,
                                        x_scale=(amp, 1, 1, 5),xtol=1e-6).x
        else:
            # old algorithm never uses multithreading. May be better.
            # out = optimize.leastsq(error, param, Dfun=jacobian)[0]
            out = optimize.leastsq(error, param, xtol=1e-6)[0]

    except Exception as err:
        # print(param)
        print('Fitting did not work: {} with initial parameters {}'.format(err, param))
        # raise err
        return param

    change = out - param
    if limit and np.abs(change[1:3]).max() >= limit:
        if verbose:
            print('Found out of limit fit result {}. Reverting to init values {}.' \
                  .format(out[1:3], param[1:3]))
        out = param

    return out


@loop_over_stack
def center_of_mass(img: np.ndarray, threshold: float = 0.0):
    """
    Returns the center of mass of an image using all the pixels larger than 
    the threshold. Automatically skips values below threshold. Fast for sparse 
    images, for more crowded ones `center_of_mass2` may be faster.
    
    Args:
        img (np.ndarray): Input image
        threshold (float, optional): minimum pixel value to include. Defaults to 0.0.
        
    Returns:
        np.ndarray: [x0, y0] -> image center of mass
    """
    cut = np.where(img>threshold)
    (y0,x0) = np.sum(img[cut]*cut,axis=1)/np.sum(img[cut])
    return np.array([x0,y0])


def center_of_mass2(img: np.ndarray, threshold: Optional[float] = None):
    """
    Returns the center of mass of an image using all the pixels larger than 
    the threshold. Automatically skips values below threshold. Can be faster
    than `center_of_mass` for crowded images (just try it out).
    
    Args:
        img (np.ndarray): Input image
        threshold (float, optional): minimum pixel value to include. If None,
            does not apply a threshold. Defaults to None.
        
    Returns:
        np.ndarray: [x0, y0] -> image center of mass
    """
    vec = np.stack(np.meshgrid(np.arange(0, img.shape[-1]), 
                               np.arange(0, img.shape[-2])), axis=-1)

    if threshold is not None:
        imgt = np.where(img >= threshold, img, 0)
    else:
        imgt = img

    com = (np.tensordot(imgt, vec, 2)
            /imgt.sum(axis=(-2, -1)).reshape(-1, 1))

    return com


@loop_over_stack
def apply_virtual_detector(img: np.ndarray, r_inner: float, r_outer: float, 
                           x0: Optional[float] = None, y0: Optional[float] = None) -> float:
    """
    Apply a "virtual STEM detector" to stack, with given inner and outer radii. Returns the mean value of all pixels
    that fall inside this annulus.

    Args:
        img (np.ndarray): input image (or stack thereof)
        r_inner (float): Inner radius
        r_outer (float): Outer radius
        x0 (float): Beam center position along x. If None, assumes center of image. Defaults to None. Should follow
            CXI convention, i.e. relative to pixel center, not corner.
        y0 (float): Similar for y

    Returns:
        float: mean value of pixels inside the annulus defined by r_inner and r_outer
    """
    ysize, xsize = img.shape
    x0 = xsize/2 - 0.5 if x0 is None else x0
    y0 = ysize/2 - 0.5 if y0 is None else y0
    x = np.arange(xsize) - x0
    y = np.arange(ysize) - y0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    # print(r_inner, r_outer)
    mask = ((R < r_outer) & (R >= r_inner)) & ((img >= 0) if (img.dtype == np.integer) else np.isfinite(img))
    # print(mask.sum())
    
    return img[mask].mean()


@loop_over_stack
def get_peaks(img: np.ndarray, x0: float, y0: float, max_peaks: int = 500, 
              pxmask: Optional[np.ndarray] = None, min_snr: float = 4., threshold: float = 7.,
              min_pix_count: int = 2, max_pix_count: int = 20, local_bg_radius: int = 3,
              min_res: int = 0, max_res: int = 500, 
              as_dict: bool = True, extended_info: bool = False) -> Union[dict,np.ndarray]:
    """Find peaks in diffraction pattern using the peakfinder8 algorithm as used in
    CrystFEL, OnDA and Cheetah. For explanation of the finding parameters, please consult the 
    CrystFEL documentation (or just run `man indexamajig`).

    Args:
        img (np.ndarray): image stack
        x0 (float): image stack x center
        y0 (float): image stack y center
        max_peaks (int, optional): maximum number of peaks. Defaults to 500.
        pxmask (Optional[np.ndarray], optional): pixel mask. Defaults to None.
        min_snr (float, optional): minimum peak SNR. Defaults to 4..
        threshold (float, optional): count threshold. Defaults to 8.
        min_pix_count (int, optional): minimum number of pixels in peak. Defaults to 2.
        max_pix_count (int, optional): maximum number of pixels in peak. Defaults to 20.
        local_bg_radius (int, optional): radius for peak backgroud estimation. Defaults to 3.
        min_res (int, optional): minimum resolution (= radial range) in pixels. Defaults to 0.
        max_res (int, optional): maximum resolution (= radial range) in pixels. Defaults to 500.
        as_dict (bool, optional): return results as a dictionary instead of a 
            single numpy array. Defaults to True.

    Returns:
        dict: CXI-format peaks information. If as_dict=False, instead returns a 1d array
            of size (3 * max_peaks + 1), which contains x positions, y positions, intensities,
            and number of peaks concatenated.
            
    Note:
        The returned peak positions follow CXI convention, that is, they refer to pixel *centers*,
        not corners (as in `CrystFEL`). For `CrystFEL`-convention you have to add 0.5 to the
        returned peak positions.
    """
    
    from .peakfinder8_extension import peakfinder_8

    X, Y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    R = (((X-x0)**2 + (Y-y0)**2)**.5).astype(np.float32)
     
    mask = np.ones_like(img, dtype=np.int8) if pxmask is None else (pxmask == 0).astype(np.int8)
    mask[R > max_res] = 0
    mask[R < min_res] = 0
           
    pks = peakfinder_8(max_peaks, 
                             img.astype(np.float32), 
                             mask, 
                             R, 
                             img.shape[1], 
                             img.shape[0], 
                             1, 
                             1, 
                             threshold, 
                             min_snr, 
                             min_pix_count, 
                             max_pix_count, 
                             local_bg_radius)
    
    fill = [0]*(max_peaks-len(pks[0]))
    result = [('peakXPosRaw', np.array(pks[0] + fill)),
            ('peakYPosRaw', np.array(pks[1] + fill)),
            ('peakTotalIntensity', np.array(pks[2] + fill)),
            ('nPeaks', np.array(len(pks[0])))]
            
    if extended_info:
        result = result[:-1] + [
            ('peakIndex', np.array(pks[3] + fill)),
            ('peakNPix', np.array(pks[4] + fill)),
            ('PeakMaxIntensity', np.array(pks[5] + fill)),
            ('PeakSigma', np.array(pks[6] + fill)),
            ('PeakSNR', np.array(pks[7] + fill)),
            ('nPeaks', np.array(len(pks[0])))
        ]
        
    if as_dict:
        return dict(result)
        # return result
    else:
        return np.array(pks[0] + fill + pks[1] + fill + pks[2] + fill + [len(pks[0])])


@loop_over_stack
def radial_proj(img: np.ndarray, x0: Optional[float] = None, y0: Optional[float] = None, 
                scale: float = 1, scale_axis: float = 0,
    my_func: Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]] = np.nanmean, 
    min_size: int = 600, max_size: int = 850, filter_len: int = 1) -> np.ndarray:
    """ 
    Applies a function to azimuthal bins of the image around 
    the center (x0, y0) for each integer radius and returns the result 
    in a np.array of size max_size, yielding a radial profile. Skips values that are set to -1 or nan.
    
    Optionally, a median filter can be applied to the output.

    Args:
        img (np.ndarray): input image or stack
        x0 (Optional[float], optional): x center of pattern. Center of image is None. Defaults to None.
        y0 (Optional[float], optional): y center of pattern. Center of image is None. . Defaults to None.
        my_func (Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]], optional): function
            to call on all pixel values at a given radius, or iterable thereof. Defaults to np.nanmean.
        min_size (int, optional): Minimum length of the output profile. Defaults to 600.
        max_size (int, optional): Maximum length of the output profile. Defaults to 850.
        filter_len (int, optional): Kernel size of median filter applied after profile calculation.
        filter_len must be odd, and filtering is at the moment incompatible with multiple functions. Defaults to 1.
        
    Returns:
        np.ndarray: radial profile calculated using my_func
        
    Note:
        The median filter will currently only work, if a single function is used only! Sorry for that.
    """
    #TODO ellipticity correction?

    if isinstance(my_func, tuple) and (len(my_func) > 1) and (filter_len > 1):
        raise ValueError('radial_proj with filtering only works if a single function is used. Sorry.')

    if filter_len//2 == filter_len/2:
        raise ValueError('filter_len must be odd.')

    if not (isinstance(my_func, list) or isinstance(my_func, tuple)):
        my_func = [my_func]

    (ylen,xlen) = img.shape
    (y,x) = np.ogrid[0:ylen,0:xlen]
    #print(x0,y0)

    x0 = img.shape[1]/2 - 0.5 if x0 is None else float(x0)
    y0 = img.shape[0]/2 - 0.5 if y0 is None else float(y0)

    # fault tolerance if absurd centers are supplied
    if np.isnan(x0) or np.isnan(y0) or x0<0 or x0>=xlen or y0<0 or y0>=ylen:
        result = np.empty(max_size*len(my_func))
        result.fill(np.nan)
        return result
    
    x, y = x - x0, y - y0
    
    # ellipticity correction
    if scale != 1:
        c, s = np.cos(scale_axis), np.sin(scale_axis)
        x, y = scale*(c*x - s*y), s*x + c*y
        x, y = c*x + s*y, -s*x + c*y

    radius = (np.rint((x**2 + y**2)**0.5) # radius coordinate of each pixel
                .astype(np.int32))
    
    center = img[int(np.round(y0)),int(np.round(x0))]
    radius[np.where((img==-1) | np.isnan(img))]=0 # ignore bad pixels by setting radius to zero
    row = radius.flatten()
    col = np.arange(len(row))
    mat = sparse.csr_matrix((img.flatten(), (row, col)))

    rng = np.min([1+np.max(radius), max_size])
    size = np.max([rng, min_size])
    result = -1 * np.ones(size*len(my_func))
    fstart = np.arange(0, size*len(my_func), size)

    for r in range(1, rng):
        rbin_data = mat[r].data

        if rbin_data.size:
            result[r + fstart] = [fn(rbin_data) for fn in my_func]

    if center > -1:
        result[fstart] = [fn(center)  for fn in my_func]

    if filter_len > 1:       
        result[filter_len//2:] = median_filter(result, filter_len)[filter_len//2:]

    assert (result.size >= min_size) and (result.size <= max_size)

    return result


@loop_over_stack
def cut_peaks(img: np.ndarray, nPeaks: np.ndarray, peakXPosRaw: np.ndarray, 
              peakYPosRaw: np.ndarray, radius: int = 2, replaceval: Union[int, float, None] = None) -> np.ndarray:
    """Cuts peaks out of an image and replaces them with replaceval. Peak positions are provided in CXI format.

    This function is mainly interesting for calculation of radial profiles, ignoring Bragg peaks.

    Args:
        img (np.ndarray): Input image (or stack thereof)
        nPeaks (np.ndarray): number of peaks
        peakXPosRaw (np.ndarray): peak X positions
        peakYPosRaw (np.ndarray): peak y positions
        radius (int, optional): Radius of circle within which image values are replaced around each peak. 
            Defaults to 2.
        replaceval (Union[int, float, None], optional): Value to paint into the circles. If None,
            uses -1 on integer images and np.nan otherwise. Defaults to None.

    Returns:
        np.ndarray: Image with cut-out peaks.
    """
    #print(nPeaks)
    if replaceval is None:
        replaceval = -1 if issubclass(img.dtype.type, np.integer) else np.nan
    nPeaks = nPeaks.squeeze()
    peakXPosRaw = peakXPosRaw.squeeze()
    peakYPosRaw = peakYPosRaw.squeeze()
    #print(peakYPosRaw[:nPeaks.squeeze()])
    mask = np.zeros_like(img).astype(np.bool)
    #print(img.shape)
    mask[(peakYPosRaw[:nPeaks]).round().astype(int), (peakXPosRaw[:nPeaks]).round().astype(int)] = True
    mask = binary_dilation(mask,disk(radius),1)
    img_nopeaks = np.where(mask,replaceval,img)
    return img_nopeaks


@loop_over_stack
def strip_img(img: np.ndarray, prof: np.ndarray, 
              x0: Optional[float] = None, y0: Optional[float] = None, 
              pxmask: Optional[np.ndarray] = None, truncate: bool = False, 
              offset: Union[float, int] = 0, keep_edge_offset: bool = False, 
              replaceval: Optional[float] = None, interp: bool = True, 
              dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Subtract a radial profile from a diffraction pattern, assuming radial symmetry of the background.

    Args:
        img (np.ndarray): Input image (or stack thereof)
        prof (np.ndarray): Radial profile to be subtracted
        x0 (float, optional): Diffraction pattern center along x. If None, use the image center.
            Defaults to None.
        y0 (float, optional): Diffraction pattern center along y. If None, use the image center.
            Defaults to None.
        pxmask (Optional[np.ndarray], optional): Pixel mask to apply *after* subtraction. Defaults to None.
        truncate (bool, optional): Replace all values below the offset by replaceval. Defaults to False.
        offset (Union[float, int], optional): Offset to apply to the output image. Required if
            you want to keep positive pixel values. Defaults to 0.
        keep_edge_offset (bool, optional): [description]. Defaults to False.
        replaceval (Optional[float], optional): Replace value for pixels falling below offset. 
            Defaults to None.
        interp (bool, optional): Interpolate background pixel values, otherwise use nearest
            neighbour. Defaults to True.
        dtype (Optional[np.dtype], optional): If not None, convert output
            image to this data type. Defaults to None.

    Returns:
        np.ndarray: Image with subtracted radial profile.
    """
    #TODO ellipticity correction?
    
    x0 = img.shape[1]/2 - 0.5 if x0 is None else float(x0)
    y0 = img.shape[0]/2 - 0.5 if y0 is None else float(y0)
    
    if np.isnan(x0) or np.isnan(y0):
        return np.zeros(img.shape)

    prof = prof.flatten()   # background profile
    ylen,xlen = img.shape
    y,x = np.ogrid[0:ylen,0:xlen]

    if interpolate:
        iprof = interpolate.interp1d(range(len(prof)), prof, fill_value=0, bounds_error=False)
        radius = ((x-x0)**2 + (y-y0)**2)**0.5
        profile = np.zeros(1+np.floor(np.max(radius)).astype(np.int32))
        bkg = iprof(radius)
    else:
        radius = (np.rint(((x-x0)**2 + (y-y0)**2)**0.5)).astype(np.int32)
        profile = np.zeros(1+np.max(radius))
        comlen = min(len(profile), len(prof))
        np.copyto(profile[:comlen], prof[:comlen])
        bkg = profile[radius]

    img_out = img - bkg + offset if keep_edge_offset else img - bkg

    dtype = img_out.dtype if dtype is None else dtype

    if replaceval is None:
        replaceval = np.nan if np.issubdtype(dtype, np.floating) else -1

    if truncate:
        img_out[img_out < offset] = replaceval

    if pxmask is not None:
        img_out = correct_dead_pixels(img_out, pxmask, 'replace', 
                                      replace_val=replaceval, mask_gaps=False)

    if not dtype == img_out.dtype:
        if np.issubdtype(dtype, np.integer):
            img_out = img_out.round()
        img_out = img_out.astype(dtype)

    return img_out


@loop_over_stack
def remove_background(img: np.ndarray, x0: Optional[float] = None, y0: Optional[float] = None,
    nPeaks: Optional[np.ndarray] = None, peakXPosRaw: Optional[np.ndarray] = None, peakYPosRaw: Optional[np.ndarray] = None, 
    peak_radius=3, filter_len=5, rfunc: Callable[[np.ndarray], np.ndarray] = np.nanmean,
    pxmask=None, truncate=False,  offset=0) -> np.ndarray:
    """Combines `radial_proj`, `cut_peaks` and `strip_img` into a background-removal protocol for diffration
    patterns, assuming radial symmetry of the background.
    
    The diffraction pattern is first azimuthally integrated, excluding Bragg peaks, and the resulting radial
    profile is further smoothed. The profile is then re-projected to the full image and subtracted. This procedure
    usually works excellently well - at least, if the peak finding has been done carefully. If there are hard
    issues with peak finding, it might be worth setting rfunc=np.nanmedian.
    
    Peaks have to be provided in CXI format and convention.

    Args:
        img (np.ndarray): Input image or stack thereof
        x0 (Optional[float], optional): Diffraction pattern center along x. If None, use the image center.
            Defaults to None.
        y0 (Optional[float], optional): Diffraction pattern center along y. If None, use the image center.
            Defaults to None.
        nPeaks (Optional[np.ndarray], optional): Number of peaks. Defaults to None.
        peakXPosRaw (Optional[np.ndarray], optional): peak X positions. Defaults to None.
        peakYPosRaw (Optional[np.ndarray], optional): peak Y positions. Defaults to None.
        peak_radius (int, optional): Radius around each peak excluded from background calculation. Defaults to 3.
        filter_len (int, optional): Range of median filter applied to radial profile. Defaults to 5.
        rfunc (Callable[[np.ndarray], np.ndarray], optional): Function for calculation of the radial profile
            through azimuthal averaging. Defaults to np.nanmean.
        pxmask ([type], optional): Pixel mask to be applied after correction. Defaults to None.
        truncate (bool, optional): Set all pixels of value < offset to 0. Defaults to False.
        offset (int, optional): Offset for the output image. Defaults to 0.

    Returns:
        np.ndarray: [description]
    """

    if np.issubdtype(img.dtype, np.integer) and offset == 0:
        warn('Removing background on an integer image with zero offset will likely cause trouble later on.')

    replace_val = np.nan if np.issubdtype(img.dtype, np.floating) else -1

    x0 = img.shape[1]/2 - .5 if x0 is None else x0
    y0 = img.shape[0]/2 - .5 if y0 is None else y0

    pxmask = ((img == np.nan) | (img == -1)) if pxmask is None else pxmask
    #print((pxmask == 0).sum())

    # print(nPeaks)

    if (nPeaks is not None) and (nPeaks > 0):
        img_nopk = cut_peaks(img, nPeaks, peakXPosRaw, peakYPosRaw, radius=peak_radius, replaceval=replace_val)
    else:
        img_nopk = img.copy()
    
    # ALWAYS mask gaps for the background determination
    # TODO THIS FAILS FOR IMAGES NOT MATCHING THE MAIN DETECTOR GEOMETRY
    img_nopk = correct_dead_pixels(img_nopk, pxmask, mask_gaps=True, strategy='replace')
    # return img_nopk
    r0 = radial_proj(img_nopk, x0, y0, my_func=rfunc, filter_len=filter_len)

    img_nobg = strip_img(img, prof=r0, x0=x0, y0=y0, pxmask=pxmask, truncate=truncate, 
        keep_edge_offset=True, interp=True, dtype=img.dtype)

    return img_nobg


@jit(['int32[:,:](int32[:,:], float64, float64, int64, int64, int64)',
      'int16[:,:](int16[:,:], float64, float64, int64, int64, int64)',
      'int64[:,:](int64[:,:], float64, float64, int64, int64, int64)',
      'float64[:,:](float64[:,:], float64, float64, int64, int64, float64)',
      'float32[:,:](float32[:,:], float64, float64, int64, int64, float64)'],
     nopython=True, nogil=True)  # ahead-of-time compilation using numba. Otherwise painfully slow.
def _center_sgl_image(img, x0, y0, xsize, ysize, padval):
    """Shifts a *single* image (not applicable to stacks!), such that the original image coordinates x0, y0 
    are in the center of the output image, which has a size of xsize, ysize.
    
    This function is typically used to change diffraction images such that the zero-order beam sits in the
    center of the image. The size of the output image should be sufficiently larger as to not truncate
    the shifted diffraction pattern.
    
    Note:
        The coordinates in this function refer to pixel centers (CXI convention), *not* pixel corners
        (CrystFEL convention). I.e., if shifting based on CrystFEL output or similar, the shifts
        must be increased by 0.5.

    Args:
        img (np.ndarray): Input image
        x0 (float): x position in input image to be shifted to the center of the output image
        y0 (float): y position in input image to be shifted to the center of the output image
        xsize (int): x size of the output image
        ysize (int): y size of the output image
        padval (float or int): value of the pixels used to pad the output image.

    Returns:
        np.ndarray: output image of size (ysize, xsize) with centered diffraction pattern
    """

    simg = np.array(padval).astype(img.dtype) * np.ones((ysize, xsize), dtype=img.dtype)
    #int64=np.int64
    #x0 -= 0.5
    xin = np.ceil(np.array([-xsize / 2, xsize / 2]) + x0, np.empty(2)).astype(int64)  # initial coordinate system
    xout = np.array([0, simg.shape[1]], dtype=int64)  # now start constructing the final coordinate system
    if xin[0] < 0:
        xout[0] = -xin[0]
        xin[0] = 0
    if xin[1] > img.shape[1]:
        xout[1] = xout[1] - (xin[1] - img.shape[1])
        xin[1] = img.shape[1]

    yin = np.ceil(np.array([-ysize / 2, ysize / 2]) + y0, np.empty(2)).astype(int64)
    yout = np.array([0, simg.shape[0]], dtype=int64)
    if yin[0] < 0:
        yout[0] = -yin[0]
        yin[0] = 0
    if yin[1] > img.shape[0]:
        yout[1] = yout[1] - (yin[1] - img.shape[0])
        yin[1] = img.shape[0]
    #print(xin,xout,yin,yout)
    simg[yout[0]:yout[1], xout[0]:xout[1]] = img[yin[0]:yin[1], xin[0]:xin[1]]

    return simg


def center_image(imgs: Union[np.ndarray, da.Array], x0: Union[np.ndarray, da.Array], 
                 y0: Union[np.ndarray, da.Array], xsize: int, ysize: int, 
                 padval: Union[float, int, None] = None, parallel: bool = True):
    """
    Shifts a stack of images, such that the original image coordinates x0, y0 
    are in the center of the output image, which has a size of xsize, ysize.
    
    This function is typically used to change diffraction images such that the zero-order beam sits in the
    center of the image. The size of the output image should be sufficiently larger as to not truncate
    the shifted diffraction pattern.
    
    Note:
        The coordinates in this function refer to pixel centers (CXI convention), *not* pixel corners
        (CrystFEL convention). I.e., if shifting based on CrystFEL output or similar, the shifts
        must be increased by 0.5.

    Args:
        imgs (Union[np.ndarray, da.Array]): Input image stack
        x0 (Union[np.ndarray, da.Array]): x position in input image to be shifted to the center of the output image
        y0 (Union[np.ndarray, da.Array]): y position in input image to be shifted to the center of the output image
        xsize (int): x size of the output image
        ysize (int): y size of the output image
        padval (Union[float, int, None], optional): value of the pixels used to pad the output image. 
            If None, use nan for float images and -1 for integer images. Defaults to None.
        parallel (bool, optional): execute operation in parallel. Defaults to True.

    Returns:
        Union[np.ndarray, da.Array]: output image stack of size (ysize, xsize) with centered diffraction patterns
    """
    
    if padval is None:
        padval = np.nan if not issubclass(imgs.dtype.type, np.integer) else -1
        print('Padding with value ', padval)
    
    if isinstance(imgs, da.Array):
        # Preprocess arguments and call function again, using map_blocks along the stack direction
        x0 = x0.reshape(-1, 1, 1)
        y0 = y0.reshape(-1, 1, 1)
        if not isinstance(x0, da.Array):
            x0 = da.from_array(x0, (imgs.chunks[0], 1, 1))
        if not isinstance(y0, da.Array):
            y0 = da.from_array(y0, (imgs.chunks[0], 1, 1))

        return imgs.map_blocks(center_image, x0, y0, xsize, ysize, padval,
                               chunks=(imgs.chunks[0], ysize, xsize),
                               dtype=imgs.dtype, parallel=False)

    # condition the input arguments a bit...
    x0 = x0.reshape(-1)
    x0[np.isnan(x0)] = imgs.shape[2] / 2
    y0 = y0.reshape(-1)
    y0[np.isnan(y0)] = imgs.shape[1] / 2
    simgs = np.array(padval).astype(imgs.dtype) * np.ones((imgs.shape[0], ysize, xsize), dtype=imgs.dtype)

    if parallel:
        it = prange(imgs.shape[0]) # uses numba's prange for parallelization
    else:
        it = range(imgs.shape[0])

    for ii in it:
        # print(x0[ii], y0[ii])
        simg = _center_sgl_image(imgs[ii, :, :], x0[ii], y0[ii], xsize, ysize, padval)
        simgs[ii, :, :] = simg

    return simgs


def apply_saturation_correction(img: np.ndarray, exp_time: float, dead_time: float = 1.9e-3, 
                                gap_factor: float = 2):
    """Apply detector correction function to image. Should ideally be done even before flatfield.
    Uses a 5th order polynomial approximation to the Lambert function, which is appropriate
    for a paralyzable detector, up to the point where its signal starts inverting (which is where
    nothing can be done anymore)
    
    The default dead time value of 1.9 microseconds has been determined for a Medipix3 sensor.
    
    Args:
        img (np.ndarray): Input image or image stack
        exp (float): Exposure time in ms
        dead_time (float, optional): Dead time of detector in ms. Defaults to 1.9e-3.
        gap_factor (float, optional): Factor to scale dead time for gap pixels. Defaults to 2.4.
    """
    lambert = lambda x: x - x**2 + 3/2*x**3 - 8/3*x**4 + 125/24*x**5
    satcorr = lambda y, sat: -lambert(-sat*y)/sat # saturation parameter: dead time/exposure time
    if gap_factor != 1:
        dt = dead_time * (1 + (gap_factor-1)*gap_pixels())
    else:
        dt = dead_time
        
    return satcorr(img, dt/exp_time)


def apply_flatfield(img: Union[np.ndarray, da.Array], reference: Union[np.ndarray, str], keep_type: bool = True, 
                    ref_smooth_range: Optional[float] = None,  
                    normalize_reference: bool = False) -> Union[np.ndarray, da.Array]:
    """Corrects the detector response by dividing the images in the image (stack) by a reference
    image (gain reference image), which should vary around 1.
    
    Args:
        img (Union[np.ndarray, da.Array]): Input image
        reference (Union[np.ndarray, str]): array containing the reference image, or filename of
            a TIF file containing the reference image
        keep_type (bool, optional): Keep the image data type, that is, round the pixel values
            back to integers if the input is an integer image. If False, the output image
            will always be a float. Defaults to True.
        ref_smooth_range (Optional[float], optional): If not None, applies a Gaussian blur to the
            reference image before correction, use this parameter to set its width. Defaults to None.
        normalize_reference (bool, optional): Re-normalize the reference image such that its
            average value is exactly 1. Defaults to False.

    Returns:
        np.ndarray: flatfield-corrected image
    """

    if isinstance(reference, str):
        reference = imread(reference).astype(np.float32)
    elif isinstance(reference, np.ndarray):
        reference = reference.astype(np.float32)
    else:
        raise TypeError('reference must be either numpy array or TIF filename')

    if normalize_reference:
        reference = reference/np.nanmean(reference)

    if ref_smooth_range is not None:
        reference = convolve(reference, Gaussian2DKernel(ref_smooth_range),
                             boundary='extend', nan_treatment='interpolate')

    if len(img.shape) > 2:
        reference = reference.reshape((1,reference.shape[-2],reference.shape[-1]))

    if keep_type:
        return (img/reference).astype(img.dtype)
    else:
        return img/reference


def correct_dead_pixels(img: Union[np.ndarray, da.Array], pxmask: Union[np.ndarray, str], 
                        strategy: str = 'interpolate', 
                        interp_range: int = 1, replace_val: Union[float, int] = None, 
                        mask_gaps: bool = False, edge_mask_x: Union[int, Tuple] = (100, 30), 
                        edge_mask_y: Union[int, Tuple] = 0, invert_mask: bool = False) -> np.ndarray:
    """Corrects a set of images for dead pixels by either replacing values with a 
    constant, or interpolation from a Gaussian-smoothed version of the image. It 
    requires a binary array (pxmask) which is 1 (or 255 or True) for dead pixels. 
    The function accepts a 3D array where the first dimension corresponds to a stack/movie.

    Args:
        img (np.ndarray): the image or image stack (first dimension is stack). 
            For strategy=='replace' it can be a dask or numpy array, otherwise numpy only.
        pxmask (Union[np.ndarray, str]): pixel mask with values as described above, or name of
            a TIF file containing the pixel mask
        strategy (str, optional): 'interpolate' or 'replace'. Defaults to 'interpolate'.
        interp_range (int, optional): range of interpolation for 'interpolate' strategy, in pixels. 
            Defaults to 1.
        replace_val (Union[float, int], optional): replacement value for 'replace' strategy. If None, use
            -1 for integer images and nan for float images. Defaults to None.
        mask_gaps (bool, optional): mask gaps between detector panels as returned by the gap_pixels() function. 
            Defaults to False.
        edge_mask_x (int, optional): Declare this number of pixels near the edges along x as
            invalid and replace them with replaceval. Defaults to 70.
        edge_mask_y (int, optional): Declare this number of pixels near the edges along y as
            invalid and replace them with replaceval. Defaults to 0.
        invert_mask (bool, optional): invert the pixel mask, i.e., invalid pixels are zero/False. Defaults to False.

    Returns:
        np.ndarray: dead-pixel corrected image. Can be da.Array for 'replace' strategy.
    """

    assert strategy in ('interpolate', 'replace')
    
    if replace_val is None:
        replace_val = -1 if isinstance(img, np.integer) else np.nan

    if isinstance(pxmask, str):
        pxmask = imread(pxmask)
    elif isinstance(pxmask, np.ndarray) or isinstance(pxmask, da.Array):
        pxmask = pxmask.astype(np.bool)
        if invert_mask:
            pxmask = np.logical_not(pxmask)
    else:
        raise TypeError('pxmask must be either Numpy array, or TIF file name')

    if mask_gaps:
        pxmask[gap_pixels()] = True

    if edge_mask_x:
        if isinstance(edge_mask_x, int):
            rng = (edge_mask_x, edge_mask_x)
        else:
            rng = edge_mask_x
            # print(rng)
        pxmask[:, :rng[0]] = True
        pxmask[:, -rng[1]:] = True

    if edge_mask_y:
        if isinstance(edge_mask_y, int):
            rng = (edge_mask_y, edge_mask_y)
        else:
            rng = edge_mask_y   
            # print(rng)     
        pxmask[:rng[0],:] = True
        pxmask[-rng[1]:,:] = True
        
    if strategy == 'interpolate':

        if (img.ndim > 2) and strategy == 'interpolate':
            return np.stack([correct_dead_pixels(theImg, pxmask=pxmask, 
                                                 strategy='interpolate', interp_range=interp_range,
                                                 replace_val=replace_val) for theImg in img])

        kernel = Gaussian2DKernel(interp_range)
        with catch_warnings():
            simplefilter("ignore")
            img_flt = convolve(img.astype(float), kernel, boundary='extend', 
                            nan_treatment='interpolate', mask=pxmask)
        
        if isinstance(img, np.integer):
            img_flt = np.nan_to_num(img_flt, copy=False, nan=-1).astype(np.int32)
        
        img_out = np.where(pxmask, img_flt, img)

        return img_out

    elif strategy == 'replace':

        if isinstance(img, np.ndarray):
            if img.ndim > 2:
                # putmask does not support broadcasting
                np.putmask(img, np.broadcast_to(pxmask, img.shape), 
                           replace_val)
            else:
                np.putmask(img, pxmask, replace_val)

            return img

        elif isinstance(img, da.Array):
             #dask arrays are immutable. This requires a slightly different way
            sz = pxmask.shape
            pml = da.from_array(pxmask.reshape(1, sz[-2], sz[-1]), 
                                chunks=(1, sz[-2], sz[-1]))
            pml = da.broadcast_to(pml, img.shape, chunks=img.chunks)

            return da.where(pml, replace_val, img)
    