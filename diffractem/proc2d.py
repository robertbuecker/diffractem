import tifffile
import numpy as np
import dask.array as da
from numba import jit, prange, int64
from . import gap_pixels
from scipy import optimize, sparse, special, interpolate
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import disk
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage.filters import median_filter
from functools import wraps
from typing import Optional, Tuple, Union, List, Callable
from warnings import warn


def loop_over_stack(fun):
    """
    Decorator to (sequentially) loop a 2D processing function over a stack. 
    Works on all functions with signature fun(imgs, *args, **kwargs), where 
    imgs is a 3D stack or a 2D single image. 
    If any of the positional/named arguments is an iterable of the same length 
    as the image stack, it is distributed over the function calls for each 
    image.
    
    :param fun     : function to be decorated
    :return         : decorated function

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
            return np.expand_dims(fun(imgs.squeeze(), *args, **kwargs), axis=0)

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
            return np.stack(out)
        except ValueError as err:
            print('Function',fun,'failed for output array construction.')
            raise err

    return loop_fun


def mean_clip(c,sigma=2.0):
    """
    Iteratively keeps only the values from the array that satisfies
            0 < c < c_mean + sigma*std 
    and return the mean of the array. Assumes the
    array contains positive entries, 
    if it does not or the array is empty returns -1 
    
    :param vector   : input vector of values.
    :param sigma    : number of standard deviations away from the mean 
                    : that is allowed.
    :return c_mean  : mean of clipped values
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


def func_lorentz(p, x, y):
    """
    Function that returns a Student't distribution or generalised Cauchy
    distribution in Two Dimensions(x,y).
    :param p    :  [amp x_0 y_0 scale shape]
    :param x    : x coordinate
    :param y    : y coordinate
    """
    return p[0]*((1+((x-p[1])/p[3])**2.0 + ((y-p[2])/p[3])**2.0)**(-p[4]/2.0))


def func_voigt(p, r):
    """
    Function that returns the voight-profile
    :param p    : [amp sigma lambda]
    :param r    : np.array of radius values to evaluate the function at
    :return     : voight_func evaluation at points r
    """
    z = (r + 1j*p[2])/(2*p[1]*np.pi)
    return np.real(p[0]*special.wofz(z)/np.sqrt(2*np.pi)/p[1])


@loop_over_stack
def lorentz_fit_simple(profile, bin_min=3, bin_max = 40, amp=500.0, scale=5.0, shape=2.0):
    """
    Simplified Lorentz fit in the 1D-radial.
    """
    assert isinstance(profile, np.ndarray)

    # section to handle arrays of profiles
    if profile.ndim == 2:
        xall = [lorentz_fit_simple(x, bin_min, bin_max, amp, scale, shape) for x in profile]
        return np.stack(xall)

    param = np.array([amp, scale, shape])
    profile=profile.squeeze()
    if np.isnan(profile).any() or (np.max(profile) < 10):
        return np.array([1, 0, 0])
    def jac_lorenz_rad(p,r,counts):
        fun = (1.0+(r/p[1])**2.0)
        d_amp = fun**(-p[2]/2.0)
        d_scale =p[0]*(r**2.0)*p[2]*(p[1]**-3.0)*(fun**(-p[2]/2.0-1.0))
        d_shape = -0.5*p[0]*(fun**(-p[2]/2.0))*np.log(fun)
        return np.transpose([d_amp, d_scale, d_shape]/np.sqrt(counts))
    def func_lorenz_rad(p,r):
        return p[0]*(1.0+(r/p[1])**2.0)**(-p[2]/2.0)
    def func_error(p, r, counts):
        return (func_lorenz_rad(p,r) - counts)/np.sqrt(counts)
    cut = np.flatnonzero(profile>0)
    cut = cut[(cut >= bin_min) & (cut <= bin_max)]
    out = optimize.least_squares(func_error,param, jac_lorenz_rad, 
                                 max_nfev=1000, args=(cut,profile[cut]),
                                 bounds=([1.0, 1.0, 1.0],np.inf))
    return out.x


def apply_virtual_detector(stack, r_inner, r_outer):
    """
    Apply a "virtual STEM detector" to stack, with given inner and outer radii. Returns the mean value of all pixels
    that fall inside this annulus.
    """
    _, ysize, xsize = stack.shape
    x = np.linspace(-xsize/2-0.5, xsize/2+0.5, xsize)
    y = np.linspace(-ysize/2-0.5, ysize/2+0.5, ysize)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)[np.newaxis,:,:]
    mask = ((R < r_outer) & (R >= r_inner)) | (stack >= 0)
    return da.where(mask, stack, 0).mean(axis=(1,2))


@loop_over_stack
def lorentz_fit(img,amp = 1.0, x_0=0.0, y_0=0.0, scale=5.0,shape=2.0,
                threshold=0):    
    """
    Fits a Lorentz profile to find the centroid (x_0,y_0) of an image. 
    Build upon optimize.least_squares function  which is thread safe
    Note: least.sq is not. Analytical Jacobian has been added.
    
    :param amp      : normalisation of Lorentzian
    :param x_0      : initial x position of centroid
    :param y_0      : initial y position of centroid
    :param scale    : scale of Lorentzian
    :param shape    : shape of Lorentzian
    :return         : output of least_squares
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


def lorentz_fit_moving(img, com, update_init=True, block_id=None, 
                       verbose=False, minimum_data = 500):
    """
    Find the center positions and peak heights from Lorentz fits for a stack 
    of images, initializing each fit with the result of the previous one 
    within the stack.
    
    :param img          : Image stack
    :param com          : Center-of-mass positions as N-by-2 array
    :param update_init  : If true, tries to initialize each fit with the 
                        : results of the previous one.
    :param block_id     : Optional, just used for status message.
    :return             : Numpy array of the fit results
    """
    out = []
    fres= []
    success = False
    np.set_printoptions(precision=2,suppress=True)

    if com.ndim == 3:
        # this case happens if function is called through a map_blocks
        com = com.squeeze(axis=2)

    if verbose:
        print(com)

    for ii, (im, c) in enumerate(zip(img, com)):

        try:

            if (not update_init) or not success:
                x0 = [im.max(), c[0], c[1], 5, 2]
            else:
                x0 = fres.x

            if verbose:
                print('Block {}, Img {} \n in {}'\
                      .format(block_id, ii, x0), end='')

            if im.max() < minimum_data:
                success = False
                raise ValueError('Not enough values')


            fres = lorentz_fit(im, amp=x0[0], x_0=x0[1], y_0=x0[2],
                               scale=x0[3], shape=x0[4], threshold=10)
            hes = np.matmul(np.transpose(fres.jac),fres.jac).flatten()
            out.append(np.append(fres.x,hes))
            success = fres.success

        except Exception as err:
            success = False
            dummy = np.empty(30)
            dummy.fill(np.nan)
            out.append(dummy)
            if verbose:
                print('Exception during fit of block {}, img{}. skipped: {}'.format(block_id, ii, err))
            continue

        if verbose:
            print(' out {}'.format(fres.x))

    return np.stack(out)


@loop_over_stack
def lorentz_fast(img, x_0=None, y_0=None, amp=None, scale=5.0, radius=None, limit=None,
                 threshold=0, threads=True, verbose=False):
    """
    Fast Lorentzian fit for finding beam center; especially suited for refinement after a reasonable estimate
    (i.e. to a couple of pixels) has been made by another method such as truncated COM.
    Compared to the other fits, it always assumes a shape parameter 2 (i.e. standard Lorentzian with asymptotic x^-2).
    It can restrict the fit to only a small region around the initial value for the beam center, which massively speeds
    up the function. Also, it auto-estimates the intial parameters somewhat reasonably if nothing else is given.
    :param img: input image or image stack. If a stack is supplied, it is serially looped. Not accepting dask directly.
    :param x_0: estimated x beam center. If None, is assumed to be in the center of the image.
    :param y_0: analogous.
    :param amp: estimated peak amplitude. If None, is set to the 99.99% percentile of img.
    :param scale: peak HWHM estimate. Default: 5 pixels
    :param radius: radius of a box around x_0, y_0 where the fit is actually done. If None, the entire image is used.
    :param limit: If not None, the fit result is discarded if the found beam_center is further away than this value from
        the initial estimate.
    :param threshold: pixel value threshold below which pixels are ignored. Best left at 0 usually.
    :param threads: if True, uses scipy.optimize.least_squares, which for larger arrays (radius more than around 15)
        uses multithreaded function evaluation. Especially for radius < 50, this may be slower than single-threaded.
        In this case, best set to False
    :param verbose: if True, a message is printed on some occasions
    :return: numpy array of refined parameters [amp, x0, y0, scale]
    """
    if (x_0 is None) or (not np.isfinite(x_0)):
        x_0 = img.shape[1] / 2
    if (y_0 is None) or (not np.isfinite(x_0)):
        y_0 = img.shape[0] / 2
    if radius is not None:
        x1 = int(x_0 - radius)
        x2 = int(x_0 + radius)
        y1 = int(y_0 - radius)
        y2 = int(y_0 + radius)
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
            out = optimize.least_squares(error, param, jac=jacobian, loss='linear',
                                         max_nfev=1000, method='lm', verbose=0).x
        else:
            # old algorithm never uses multithreading. May be better.
            out = optimize.leastsq(error, param, Dfun=jacobian)[0]

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
def center_of_mass(img, threshold=0.0):
    """
    Returns the center of mass of an image using all the pixels larger than 
    the threshold. Automatically skips values below threshold. Fast for sparse 
    images.
    :param img              : Input image
    :param threshold        : minimum pixel value to include
    :return (x0,y0)         :Return location
    """
    cut = np.where(img>threshold)
    (y0,x0) = np.sum(img[cut]*cut,axis=1)/np.sum(img[cut])
    return np.array([x0,y0])


def center_of_mass2(img, threshold=None):
    """
    Alternative COM function, acting on 3D arrays directly, 
    including lazy dask arrays. Tends to be slower than center_of_mass for 
    sparse images with high thresholds, otherwise faster.
    :param img: 
    :return: 
    """
    vec = np.stack(np.meshgrid(np.arange(0, img.shape[-1]), 
                               np.arange(0, img.shape[-2])), axis=-1)

    if isinstance(img, np.ndarray):
        if threshold is not None:
            imgt = np.where(img >= threshold, img, 0)
        else:
            imgt = img

        com = (np.tensordot(imgt, vec, 2)
                /imgt.sum(axis=(-2, -1)).reshape(-1, 1))

    elif isinstance(img, da.core.Array):
        if threshold is not None:
            imgt = da.where(img >= threshold, img, 0)
        else:
            imgt = img
        # dask array tensordot requires a different axes convention. Whatever.
        com = (da.tensordot(imgt, vec.transpose((1, 0, 2)), 2) 
                / imgt.sum(axis=(-2, -1)).reshape(-1, 1))

    #if img.ndim < 3:
    #   com = com.squeeze()

    return com


@loop_over_stack
def radial_proj(img: np.ndarray, x0: Optional[float]=None, y0: Optional[float]=None, 
    my_func: Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]] = np.nanmean, 
    min_size: int = 600, max_size: int = 850, filter_len: int = 1):
    """
    Applies the function to the azimuthal bins of the image around 
    the center (x0,y0) for each integer radius and returns the result 
    in a np.array of size max_size. Skips values that are set to -1 or nan.
    :param img: input image
    :param x0: x center of mass of image
    :param y0: y center of mass of image
    :param my_func: function to be applied to the bins, or list thereof
    :param min_size: minimum length of output array
    :param max_size: maximum length of output array
    :param filter_len: kernel size of median filter applied after profile calculation.
        filter_len must be odd, and filtering is at the moment incompatible with multiple functions
    :return result: array of function returns on each radius.
    """
    # BUG: median filter 

    if isinstance(my_func, tuple) and (len(my_func) > 1) and (filter_len > 1):
        raise ValueError('radial_proj with filtering only works if a single function is used. Sorry.')

    if filter_len//2 == filter_len/2:
        raise ValueError('filter_len must be odd.')

    if not (isinstance(my_func, list) or isinstance(my_func, tuple)):
        my_func = [my_func]

    (ylen,xlen) = img.shape
    (y,x) = np.ogrid[0:ylen,0:xlen]
    #print(x0,y0)

    x0 = img.shape[1]/2 if x0 is None else float(x0)
    y0 = img.shape[0]/2 if y0 is None else float(y0)

    if np.isnan(x0) or np.isnan(y0) or x0<0 or x0>=xlen or y0<0 or y0>=ylen:
        result = np.empty(max_size*len(my_func))
        result.fill(np.nan)
        return result

    radius = (np.rint(((x-x0)**2 + (y-y0)**2)**0.5) # radius coordinate of each pixel
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
def cut_peaks(img: np.ndarray, nPeaks: np.ndarray, peakXPosRaw: np.ndarray, peakYPosRaw: np.ndarray, radius=2, replaceval=-1):
    """Cuts peaks out of an image and replaces them with replaceval.
    Peak positions are provided in CXI format.
    
    Arguments:
        img {np.array} -- Input image
        nPeaks {np.array} -- Number of peaks
        peakXPosRaw {np.array} -- X position of peaks
        peakYPosRaw {np.array} -- Y position of peaks
    
    Keyword Arguments:
        radius {int} -- Radius in pixel of peak cut-out region (default: {2})
        replaceval {int} -- Value to change the cut pixels to (default: {-1})
    
    Returns:
        [as img] -- Image with peaks cut out
    """
    #print(nPeaks)
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
def strip_img(img, x0, y0, prof: np.ndarray, pxmask: Optional[np.ndarray]=None, truncate=False, 
    offset=0, keep_edge_offset=False, replaceval: Optional[float]=None, interp=True, dtype=None):
    """
    Given an image, center coordinate(x0,y0) and a radial profile, removes the
    radial profile from the image.
    """
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
def remove_background(img, x0: Optional[Union[float]] = None, y0: Optional[Union[float]] = None,
    nPeaks: Optional[np.ndarray] = None, peakXPosRaw: Optional[np.ndarray] = None, peakYPosRaw: Optional[np.ndarray] = None, 
    peak_radius=3, filter_len=5, rfunc: Callable[[np.ndarray], np.ndarray] = np.nanmean,
    pxmask=None, truncate=False,  offset=0):

    if np.issubdtype(img.dtype, np.integer) and offset == 0:
        warn('Removing background on an integer image with zero offset will likely cause trouble later on.')

    replace_val = np.nan if np.issubdtype(img.dtype, np.floating) else -1

    x0 = img.shape[1]/2 if x0 is None else x0
    y0 = img.shape[0]/2 if y0 is None else y0

    pxmask = ((img == np.nan) | (img == -1)) if pxmask is None else pxmask
    #print((pxmask == 0).sum())

    #print(nPeaks)

    if (nPeaks is not None) and (nPeaks > 0):
        img_nopk = cut_peaks(img, nPeaks, peakXPosRaw, peakYPosRaw, radius=peak_radius, replaceval=replace_val)
    else:
        img_nopk = img

    r0 = radial_proj(img_nopk, x0, y0, my_func=rfunc, filter_len=filter_len)
    img_nobg = strip_img(img, x0, y0, r0, pxmask=pxmask, truncate=truncate, 
        keep_edge_offset=True, interp=True, dtype=img.dtype)

    return img_nobg


@jit(['int32[:,:](int32[:,:], float64, float64, int64, int64, int64)',
      'int16[:,:](int16[:,:], float64, float64, int64, int64, int64)',
      'int64[:,:](int64[:,:], float64, float64, int64, int64, int64)',
      'float64[:,:](float64[:,:], float64, float64, int64, int64, float64)',
      'float32[:,:](float32[:,:], float64, float64, int64, int64, float64)'],
     nopython=True, nogil=True)  # ahead-of-time compilation using numba. Otherwise painfully slow.
def _center_sgl_image(img, x0, y0, xsize, ysize, padval):
    """
    Shits a single image, such that the original image coordinates x0, y0 are in the center of the
    output image, which as a size of xsize, ysize.
    IMPORTANT NOTE: the coordinates in this function refer to pixel centers, not pixel corners
    (as e.g. CrystFELs peak positions). I.e., if shifting based on CrystFEL output or similar, the shifts
    must be increased by 0.5.
    :param img: input image (2D array, must be integer)
    :param x0: x coordinate in img to be in center of output image
    :param y0: y coordinate in img to be in center of output image
    :param xsize: x size of output image
    :param ysize: y size of output image
    :param padval: padding value of undefined pixels in output image
    :return: output image of shape (ysize, xsize)
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


def center_image(imgs, x0, y0, xsize, ysize, padval, parallel=True):
    """
    Centers a whole stack of images. See center_sgl_image for details... now, imgs is a 3D stack,
    x0 and y0 are 1d arrays. imgs can be a dask array, map_blocks is automatically invoked then
    """

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
                               dtype=imgs.dtype)

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
        simg = _center_sgl_image(imgs[ii, :, :], x0[ii], y0[ii], xsize, ysize, padval)
        simgs[ii, :, :] = simg

    return simgs


def apply_saturation_correction(img: np.ndarray, exp_time: float, dead_time: float = 1.9e-3):
    """Apply detector correction function to image. Should ideally be done even before flatfield.
    Uses a 5th order polynomial approximation to the Lambert function, which is appropriate
    for a paralyzable detector, up to the point where its signal starts inverting (which is where
    nothing can be done anymore)
    
    Arguments:
        img {np.ndarray} -- Input image or image stack
        exp {float} -- Exposure time in ms
    
    Keyword Arguments:
        dead_time {float} -- Dead time of detector in ms (default: {1.9e-3})
    """
    lambert = lambda x: x - x**2 + 3/2*x**3 - 8/3*x**4 + 125/24*x**5
    satcorr = lambda y, sat: -lambert(-sat*y)/sat # saturation parameter: dead time/exposure time

    return satcorr(img, dead_time/exp_time)


def apply_flatfield(img, reference=None, keep_type=True, ref_smooth_range=None, 
                    normalize_reference=False, **kwargs):
    """
    Applies a flatfield (gain reference) image to a single image or stack.
    :param img              : Input image or image stack
    :param reference        : Gain reference image (usually normalized to 1)
    :param keep_type        : Keep integer data type of the initial image, 
                            : even if it requires rounding
    :param ref_smooth_range : Optionally, smooth the reference image out
    :param normalize_reference
                            : Optionally, re-normalize the image
    :param kwargs           : Nothing, currently
    :return                 : Corrected image
    """

    if isinstance(reference, str):
        with tifffile.TiffFile(reference) as tif:
            reference = tif.asarray().astype(np.float32)
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


def correct_dead_pixels(img, pxmask=None, strategy='interpolate', 
                        interp_range=1, replace_val=-1, mask_gaps=False, 
                        edge_mask=70, **kwargs):
    """
    Corrects a set of images for dead pixels by either replacing values with a 
    constant (e.g. for Diffraction Analysis with mask support), or 
    interpolation from a Gaussian-smoothed version of the image (e.g. for SPA 
    or general imaging). It requires a binary array (pxmask) which is 
    1/255/True for dead pixels. The function accepts a 3D array where the first 
    dimension corresponds to a stack/movie. The convolution used for the Gauss 
    filter is taken from the astropy package, which allows to ignore NaNs. 
    While the function does support stacks, this may be slow, especially with 
    interpolation. In these cases, better apply to single images in parallel 
    (e.g. using diffractem.compute.process_stack).
    :param img      : the image or image stack (first dimension is stack) 
                    : to process. For replace strategy it can be a dask array.
    :param pxmask   : pixel mask with dimension of the image, or TIF file 
                    : containing pixel mask
    :param strategy : 'replace' with constant or 'interpolate' with smoothed 
                    : adjacent region
    :param interp_range : range over which interpolation pixels are calculated 
                    : (if strategy is 'interpolate')
    :param replace_val: value with which dead pixels are replaced 
                    : (if strategy is 'replace')
    :param mask_gaps: treat the interpolated pixels between the panels as dead
    :param edge_mask: number of pixels at outer left/right edges to treat as 
                    : dead (because of shading)
    :param kwargs   : not doing anything so far
    :return         : corrected image
    """

    assert strategy in ('interpolate', 'replace')

    if isinstance(pxmask, str):
        with tifffile.TiffFile(pxmask) as tif:
            pxmask = tif.asarray().astype(np.bool)
    elif isinstance(pxmask, np.ndarray) or isinstance(pxmask, da.core.Array):
        pxmask = pxmask.astype(np.bool)
    else:
        raise TypeError('pxmask must be either Numpy array, or TIF file name')

    if mask_gaps:
        pxmask[gap_pixels()] = True

    if edge_mask:
        pxmask[:, :edge_mask] = True
        pxmask[:, -edge_mask:] = True

    if strategy == 'interpolate':

        if (img.ndim > 2) and strategy == 'interpolate':
            return np.stack([correct_dead_pixels(theImg, pxmask=pxmask, strategy='interpolate', interp_range=interp_range,replace_val=replace_val) for theImg in img])

        img = img.copy()
        img_flt = img.astype(float)
        img_flt[pxmask] = np.nan
        kernel = Gaussian2DKernel(interp_range)
        img_flt = convolve(img_flt, kernel, boundary='extend', 
                           nan_treatment='interpolate')
        img_flt[np.isnan(img_flt)] = np.nanmedian(img_flt)
        img[pxmask] = img_flt.astype(img.dtype)[pxmask]

        return img

    elif strategy == 'replace':

        if isinstance(img, np.ndarray):
            if img.ndim > 2:
                # putmask does not support broadcasting
                np.putmask(img, np.broadcast_to(pxmask, img.shape), 
                           replace_val)
            else:
                np.putmask(img, pxmask, replace_val)

            return img

        elif isinstance(img, da.core.Array):
             #dask arrays are immutable. This requires a slightly different way
            sz = pxmask.shape
            pml = da.from_array(pxmask.reshape(1, sz[-2], sz[-1]), 
                                chunks=(1, sz[-2], sz[-1]))
            pml = da.broadcast_to(pml, img.shape, chunks=img.chunks)

            return da.where(pml, replace_val, img)
    