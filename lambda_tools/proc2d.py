import tifffile
import warnings
#import time
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
from numba import jit, prange, int64, float64

#import hyperspy.api as hs

from . import gap_pixels
from scipy import optimize, sparse, special, ndimage
from astropy.convolution import Gaussian2DKernel, convolve
from sklearn.cluster import KMeans, DBSCAN

from functools import wraps
#from itertools import repeat


def loop_over_stack(fun):
    """
    Decorator to (sequentially) loop a 2D processing function over a stack. 
    Works on all functions with signature fun(imgs, *args, **kwargs), where 
    imgs is a 3D stack or a 2D single image. 
    If any of the positional/named arguments is an iterable of the same length 
    as the image stack, it is distributed over the function calls for each 
    image.
    
    :param func     : function to be decorated
    :return         : decorated function

    """
    #TODO: handle functions with multiple outputs, and return a list of ndarrays
    #TODO: allow switches for paralellization

    @wraps(fun)
    def loop_fun(imgs, *args, **kwargs):
        # this is NOT intended for dask arrays!
        assert isinstance(imgs, np.ndarray) 

        if imgs.ndim == 2:
            return fun(imgs, *args, **kwargs)

        elif imgs.shape[0] == 1:
            return np.expand_dims(fun(imgs.squeeze(), *args, **kwargs), axis=0)

        #print('Applying {} to {} images'.format(fun, imgs.shape[0]))

        def _isiterable(arg):
            try:
                (a for a in arg)
                return True
            except TypeError:
                return False


        iter_args = []
        for a in args:
            if _isiterable(a) and len(a) == len(imgs):
                iter_args.append(a)
            else:
                #iter_args.append(repeat(a, lenb(imgs)))
                iter_args.append([a]*len(imgs))

        iter_kwargs = []
        for k, a in kwargs.items():
            if _isiterable(a) and len(a) == len(imgs):
                iter_kwargs.append(a)
            else:
                #iter_kwargs.append(repeat(a, len(imgs)))
                iter_kwargs.append([a] * len(imgs))


        out = []
        #print(iter_args)
        #print(iter_kwargs)

        for arg in zip(imgs, *(iter_args + iter_kwargs)):
            theArgs = arg[1:1+len(args)]
            theKwargs = {k: v for k, v in zip(kwargs, arg[1+len(args):])}
            #print(theArgs)
            #print(theKwargs)
            out.append(fun(arg[0], *theArgs, **theKwargs))

        return np.stack(out)

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


def func_voight(p, r):
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
    the function up. Also, it auto-estimates the intial parameters somewhat reasonably if nothing else is given.
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
    if x_0 is None:
        x_0 = img.shape[1] / 2
    if y_0 is None:
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
        print('Fitting did not work: {}'.format(err))
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
def radial_proj(img, x0, y0, my_func=np.mean, max_size=850):
    """
    Applies the function to the azimuthal bins of the image around 
    the center (x0,y0) for each integer radius and returns the result 
    in a np.array of size max_size. Skips values that are set to -1.
    This is the split-apply-combine paradigm, done with a sparse matrix.
    :param img: input image
    :param x0: x center of mass of image
    :param y0: y center of mass of image
    :param function: function to be applied to the bins
    :param max_size: size of returned np.array hyperspy requires all returned 
                        arrays to be of the same size
    :return result: array of function returns on each radius.
    TODO: allow my_func to be a list of functions, and return multiple outputs
    """

    if not (isinstance(my_func, list) or isinstance(my_func, tuple)):
        my_func = (my_func, )

    (ylen,xlen) = img.shape
    (y,x) = np.ogrid[0:ylen,0:xlen]
    if np.isnan(x0) or np.isnan(y0) or x0<0 or x0>=xlen or y0<0 or y0>=ylen:
        result = np.empty(max_size*len(my_func))
        result.fill(np.nan)
        return result

    radius = (np.rint(((x-x0.flatten())**2 + (y-y0.flatten())**2)**0.5)
                .astype(np.int32))
    center = img[int(np.round(y0)),int(np.round(x0))]
    radius[np.where(img==-1)]=0
    row = radius.flatten()
    col = np.arange(len(row))
    mat = sparse.csr_matrix((img.flatten(), (row, col)))

    size = np.min([1+np.max(radius), max_size])
    result = -1 * np.ones(size*len(my_func))
    fstart = np.arange(0, size*len(my_func), size)

    for r in range(1, size):
        rbin_data = mat[r].data

        if rbin_data.size:
            result[r + fstart] = [fn(rbin_data) for fn in my_func]

    if center > -1:
        result[fstart] = [fn(center)  for fn in my_func]

    return result


def profile_classify_ML(prof_stack, fit_record = None, show_plot = False, 
                        refit = True, rescale = True, verbose = True, 
                        bin_min = 5, bin_max = 40):
    """
    Takes the profile stack,
    refits a 1D Lorentz distribution(Optional),
    classifies the profiles with KMeans form sklearn.cluster, 
    NB MinibatchKMeans is not parrallelized
    takes the mean, and returns them to the correct index in profile_ave.
    :param prof_stack   : stack of profiles
    :fit_record         : array of parameters returned form the 2D lorentz fit
    :show_plot          : switch for plotting the profiles
    :refit              : switch for refitting of the profiles
    :rescale            : if profiles should be scaled.
    """
    #scale,shape = np.transpose([rec[0][3:5] for rec in fit_record])
    n=prof_stack.shape[0]
    amp = np.zeros(n)
    scale = np.zeros(n)
    shape = np.zeros(n)
    prof_ave = np.zeros(prof_stack.shape)

    ###(Optional) Refitting of the profiles with a Lorentz distribution
    if refit or fit_record is None:
        prof_stack = prof_stack[:,:,np.newaxis]
        amp, scale, shape = (
                np.transpose(lorentz_fit_simple(prof_stack,bin_min,bin_max)))
        prof_stack = prof_stack.squeeze()
    else:
        amp, scale, shape = np.transpose(fit_record[:,[0,3,4]])

    if rescale:
        prof_stack = np.transpose(np.transpose(prof_stack)/amp)

    ###Grouping of profiles according to scale and shape
    X = np.transpose([100*scale,100*shape])
    # n_jobs=-1 Uses all cores
    cluster = KMeans(n_jobs=-1, n_clusters=int(np.sqrt(n))).fit(X)
    labels = cluster.labels_
    unique_labels = set(labels)
    print('Grouping the radial profiles into {} groups.'
          .format(len(unique_labels)))
    if show_plot :
        fig = plt.figure(0)
        fig.clf()
        plt.scatter(X[:, 0], X[:, 1], c = labels, marker = 'o')
        plt.title('Estimated number of clusters: {}'.
                  format(len(unique_labels)))
        plt.show()

    ###Taking Average of profiles
    prof_stack[prof_stack == -1] = np.nan
    for kk in unique_labels:
        cut = (kk==labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ave = np.nanmean(prof_stack[cut],axis=0)
        ave = np.nan_to_num(ave)
        prof_ave[cut] = ave
    
    if rescale:
        prof_ave = np.transpose(np.transpose(prof_ave)*amp)
    return prof_ave, labels


def profile_classify(prof_stack, fit_record = None, show_plot = False, 
                     show_fit = True, refit = True, rescale = True, 
                     verbose = True, bin_min = 5, bin_max = 40):
    """
    Takes the profile stack,
    refits a 1D Lorentz distribution(Optional),
    groups the profiles with the same shape and scale together,
    takes the mean and std of the group,
    plots all the profiles in the group on the same figure(Optional)
    and returns them to the correct index in profile_ave and profile_std.
    :param prof_stack   : stack of profiles
    :fit_record         : array of parameters returned form the 2D lorentz fit
    :show_plot          : switch for plotting the profiles
    :refit              : switch for refitting of the profiles
    :rescale            : if profiles should be scaled.
    """
    #scale,shape = np.transpose([rec[0][3:5] for rec in fit_record])
    n=prof_stack.shape[0]
    amp = np.zeros(n)
    scale = np.zeros(n)
    shape = np.zeros(n)
    prof_ave = np.zeros(prof_stack.shape)
    prof_std = np.zeros(prof_stack.shape)

    ###(Optional) Refitting of the profiles with a Lorentz distribution
    if refit or fit_record is None:
        prof_stack = prof_stack[:,:,np.newaxis]
        amp, scale, shape = (
                np.transpose(lorentz_fit_simple(prof_stack,bin_min,bin_max)))
        prof_stack = prof_stack.squeeze()
    else:
        amp, scale, shape = np.transpose(fit_record[:,[0,3,4]])

    if rescale:
        prof_stack = np.transpose(np.transpose(prof_stack)/amp)

    ###Grouping of profiles according to scale and shape
    keys = np.rint(scale*10)*100 + np.rint(shape*10)       
    unique_keys, inverse, counts = np.unique(keys.astype(np.int32),
                                             return_inverse = True, 
                                             return_counts = True)
    prof_stack[prof_stack == -1] = np.nan
    print('Grouping the radial profiles into {} groups:'
          .format(len(unique_keys)))
    print('Categories: {}'.format(unique_keys))
    print('    Counts: {}'.format(counts))
    for kk in unique_keys:
        cut = (kk==keys)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ave = np.nanmean(prof_stack[cut],axis=0)
            std = np.nanstd(prof_stack[cut],axis=0)
        ave = np.nan_to_num(ave)
        std = np.nan_to_num(std)
        if show_plot:
            fig = plt.figure(kk)
            fig.clf()
            plt.plot(np.transpose(prof_stack[keys==kk,0:20]))
        ###(Optional) Plotting of the first 20 bins of the profiles
        prof_ave[cut] = ave
        prof_std[cut] = std
    
    if rescale:
        prof_ave = np.transpose(np.transpose(prof_ave)*amp)
        prof_std = np.transpose(np.transpose(prof_std)*amp)
    return prof_ave, prof_std, inverse, counts


@loop_over_stack
def strip_img(img, x0, y0, prof, pxmask=None, truncate=True, offset=0, dtype=np.int16):
    """
    Given an image, coordinate(x0,y0) and a radial profile, removes the
    radial profile from the image.
    """
    if np.isnan(x0) or np.isnan(y0):
        return np.zeros(img.shape)
    x0 = x0.flatten()
    y0 = y0.flatten()
    prof = prof.flatten()
    ylen,xlen = img.shape
    y,x = np.ogrid[0:ylen,0:xlen]
    radius = (np.rint(((x-x0)**2 + (y-y0)**2)**0.5)).astype(np.int32)
    profile = np.zeros(1+np.max(radius))
    np.copyto(profile[0:len(prof)], prof)
    bkg = profile[radius]

    img_out = img - bkg + offset

    if pxmask is not None:
        img_out = correct_dead_pixels(img_out, pxmask, 'replace', 
                                      replace_val=-1, mask_gaps=True)
    if truncate:
        img_out[img_out < offset] = -1

    if not dtype == img_out.dtype:
        if issubclass(dtype, np.integer) or issubclass(dtype, int):
            img_out = img_out.round()
        img_out = img_out.astype(dtype)

    return img_out


def peak_finder(img_raw, xy, profile = None, pxmask = None, noise = np.array
                ([[False,True,False],[True,True,True],[False,True,False]]), 
                 kernel_size = 9, threshold = 9, db_eps = 1.9,
                 db_samples = 9, bkg_remove = False, img_clean = True,
                 radial_min = 5):
    """
    1) (OPTIONAL) Construct background from xy and profile. 
        Subtract from img_raw.
    2) Remove Impulse noise using morpholigical opening and closing with
        noise struc.
    3) Convolve Image with Gaussian kernel to find local background. 
        Features are all pixels larger than the local background
        by a certain threshold.
    4) Pick peaks out of image features using DBSCAN (Clustering)
    5) Labelling of Peaks using ndimage
    6) (OPTIONAL) Clean image
    Simple peakfinder built upon scipy.ndimage. 
    Uses a morpholgical opening to find features larger than size and higher
    than threshold. The features are then labelled and the center of mass and 
    sum of each peak is returned in a numpy array.
    """
    ### 1) Strip Background
    if bkg_remove:
        ylen,xlen = img_raw.shape
        y,x = np.ogrid[0:ylen,0:xlen]
        radius = (np.rint(((x-xy[0])**2 + (y-xy[1])**2)**0.5)).astype(np.int32)
        prof = np.zeros(1+np.max(radius))
        np.copyto(prof[0:len(profile)], profile)
        bkg = prof[radius]
        img = img_raw - bkg
        img = correct_dead_pixels(img, pxmask, 'replace', replace_val=-1, 
                              mask_gaps=True)
    else:
        img = img_raw
        bkg=None
    ### 2) Remove Impulse Noise
    img = ndimage.morphology.grey_opening(img, structure=noise)
    img = ndimage.morphology.grey_closing(img, structure=noise)
    ### 3) Feature detection NB: astropy.convolve is slow
    img_fil = np.where(img==-1,0,img)
    img_fil = ndimage.gaussian_filter(img_fil.astype(np.float),
                                      kernel_size,mode='constant',cval=0)
    img_norm = np.where(img==-1,0,1)
    img_norm = ndimage.gaussian_filter(img_norm.astype(np.float),
                                       kernel_size,mode='constant',cval=0)
    img_norm = np.where(img_norm==0.0,1,img_norm)
    img_fil = img_fil/img_norm
    img_feat = img - img_fil
    ### 4) Peak Picking
    loc = np.where(img_feat > threshold)
    X = np.transpose(np.stack([loc[1], loc[0] , np.log(img_feat[loc])]))
    db = DBSCAN(eps=db_eps, min_samples=db_samples, n_jobs=-1).fit(X)
    img_label = np.zeros(img.shape)
    img_label[loc] = 1 + db.labels_
    num_feat = len(set(db.labels_))
    ### 5) Peak Labelling
    com = ndimage.center_of_mass(img_feat,img_label,np.arange(1, num_feat))
    vol = ndimage.sum(img_feat,img_label,np.arange(1, num_feat))
    ### 6) Apply radial cut to peaks
    rad = np.sqrt(np.sum(np.square(np.array(com)-xy[::-1]),axis=1))
    rad_cut = rad > radial_min
    com = np.array(com)[rad_cut]
    vol = vol[rad_cut]
    img_label[img_label-1 == np.where(rad<radial_min)] = 0
    print("Found {} peaks".format(len(com)))
    ### 7) Clean Image
    if img_clean:
        img[img_label==0] = 0
    return np.column_stack((com,vol)), img


@jit(['int32[:,:](int32[:,:], float64, float64, int64, int64, int64)',
      'int16[:,:](int16[:,:], float64, float64, int64, int64, int64)',
      'int64[:,:](int64[:,:], float64, float64, int64, int64, int64)'],
     nopython=True, nogil=True)  # ahead-of-time compilation using numba. Otherwise painfully slow.
def center_sgl_image(img, x0, y0, xsize, ysize, padval):
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
        simg = center_sgl_image(imgs[ii, :, :], x0[ii], y0[ii], xsize, ysize, padval)
        simgs[ii, :, :] = simg

    return simgs


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
        reference = reference.reshape(1,reference.shape[-2],reference.shape[-1])

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
    (e.g. using lambda_tools.compute.process_stack).
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
    