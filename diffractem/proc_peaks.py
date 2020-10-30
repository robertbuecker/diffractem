# Friedel-pair refinement
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import current_process
from typing import Optional
from .pre_proc_opts import PreProcOpts
from . import tools, proc2d
from warnings import warn


def _ctr_from_pks(pkl: np.ndarray, p0: np.ndarray,
                  int_weight: bool = False, sigma: float = 2.0, bound: float = 5.0, label: str = None):
    """Gets the refined peak center position from a list of peaks containing Friedel mates
    
    Arguments:
        pkl {np.ndarray} -- [List of peaks, with x and y values in 0th and 1st column, optionally intensity on 2nd]
        p0 {np.ndarray} -- [Initial position]
    
    Keyword Arguments:
        int_weight {bool} -- [weight peaks by their intensity] (default: {False})
        sigma {float} -- [assumed peak rms radius for matching] (default: {2.0})
        bound {float} -- [maximum shift] (default: {5.0})
        label {str} -- [label to be returned in output] (default: {None})
    
    Returns:
        [tuple] -- [refined position, inverse cost function, label]
    """
    if int_weight:
        corr = lambda p: np.sum(np.matmul(pkl[:, 2:3], pkl[:, 2:3].T)
                                * np.exp(-((pkl[:, 0:1] + pkl[:, 0:1].T - 2 * p[0]) ** 2
                                           + (pkl[:, 1:2] + pkl[:, 1:2].T - 2 * p[1]) ** 2) / (2 * sigma ** 2))) \
                         / np.sum(np.matmul(pkl[:, 2:3], pkl[:, 2:3].T))
    else:
        corr = lambda p: np.sum(np.exp(-((pkl[:, 0:1] + pkl[:, 0:1].T - 2 * p[0]) ** 2
                                         + (pkl[:, 1:2] + pkl[:, 1:2].T - 2 * p[1]) ** 2) / (2 * sigma ** 2))) \
                         / (2*pkl.shape[0])

    fun = lambda p: 1 / max(corr(p), 1e-10)  # prevent infs
    if np.isnan(fun(p0)):
        return p0, np.nan, label
    else:
        lsq = least_squares(fun, p0, bounds=(p0 - bound, p0 + bound))
        return lsq.x - 0.5, 1 / lsq.cost, label # -0.5 changes from CrystFEL-like to pixel-center convention


def center_friedel(peaks: pd.DataFrame, shots: Optional[pd.DataFrame] = None, 
                    p0=(778, 308), colnames=('fs/px', 'ss/px'), sigma=2,
                   minpeaks=4, maxres: Optional[float] = None):
    """[Center refinement of diffraction patterns from a list of peaks, assuming the presence
        of a significant number of Friedel mates.]
    
    Arguments:
        peaks {[pd.DataFrame]} -- [peaks list for entire data set, as returned by StreamParser. CrystFEL convention!]
    
    Keyword Arguments:
        shots {[pd.DataFrame]} -- [shot list of data set, optional] (default: {None})
        p0 {tuple} -- [starting position for center search] (default: {(778, 308)})
        colnames {tuple} -- [column names for x and y coordinate] (default: {('fs/px', 'ss/px')})
        sigma {int} -- [peak rms radius (determines 'sharpness' of matching)] (default: {2})
        minpeaks {int} -- [minimum peak number to try matching] (default: {4})
        maxres {int} -- [maximum radius of peaks to still be considered] (default: {None})
    """
    colnames = list(colnames)
    p0 = np.array(p0)

    if current_process().daemon:
        print('Danger, its a Daemon.')

    with ProcessPoolExecutor() as p:
        futures = []
        for grp, pks in peaks.groupby(['file', 'Event']):
            pkl = pks.loc[:, colnames].values
            rsq = (pkl[:, 0] - p0[0]) ** 2 + (pkl[:, 1] - p0[1]) ** 2
            if maxres is not None:
                pkl = pkl[rsq < maxres ** 2, :]
            if (minpeaks is None) or pkl.shape[0] > minpeaks:
                futures.append(p.submit(_ctr_from_pks, pkl, p0, sigma=sigma, label=grp))

    wait(futures, return_when=ALL_COMPLETED)
    if len(futures) == 0:
        cpos = shots[['file', 'Event']].copy()
        cpos['beam_x'] = p0[0]
        cpos['beam_y'] = p0[1]
        cpos['friedel_cost'] = np.nan

        return cpos

    # reformat result into a dataframe
    cpos = pd.concat([pd.DataFrame(data=np.array([t.result()[2] for t in futures if t.exception() is None]),
                                   columns=['file', 'Event']),
                      pd.DataFrame(data=np.array([t.result()[0] for t in futures if t.exception() is None]),
                                   columns=['beam_x', 'beam_y']),
                      pd.DataFrame(data=np.array([t.result()[1] for t in futures if t.exception() is None]),
                                   columns=['friedel_cost'])],
                     axis=1)

    if shots is not None:
        # include shots that were not present in the peaks table
        cpos = shots[['file', 'Event']].merge(cpos, on=['file', 'Event'], how='left'). \
            fillna({'beam_x': p0[0], 'beam_y': p0[1]})

    return cpos


def get_acf(npk, x, y, I=None, roi_length=512, output_radius=256, 
            oversample=4, radial=True, px_ang=None, execution='processes'):
    """Gets the autocorrelation/pair correlation function of Bragg peak positions, 
    optionally with intensity weighting.
    
    It is important to set the computation region properly (i.e., the
    maximum peak positions from the center to take into account), as this affects
    computation speed and impact of non-paraxiality at larger angles. It can
    be defined using the `roi_length` argument.
    
    Peaks must be given in CXI format.
    
    Args:
        npk (np.ndarray, int): number of peaks
        x (np.ndarray): x-coordinates of peaks, *relative to pattern center*
        y (np.ndarray): y-coordinates of peaks, *relative to pattern center*
        I ([type], optional): peak intensities. Set to 1 if None. Defaults to None.
        roi_length (int, optional): edge length of the region around the image
            center that is used for the computation. Defaults to 512.
        output_radius (int, optional): maximum included radius of the output ACF. 
            The size of the 2D output will be 2*output_radius*oversample, 
            the size of the radial average will be output_radius*oversample. Defaults to 600.
        oversample (int, optional): oversampling, that is, by how much smaller the bin
            sizes of the output are than that of the input (usually the pixels). Defaults to 4.
        radial (bool, optional): compute the radial average of the ACF. Defaults to True.
        px_ang (double, optional): diffraction angle corresponding to a distance of 1 pixel
            from the center, given in rad (practically: detector pixel size/cam length). If
            given, non-paraxiality of the geometry is corrected (not tested well yet).
            Defaults to None.
        execution (str, optional): way of parallelism if a stack of pattern peak data
            is supplied. Can be 'single-threaded', 'threads', 'processes'.

    Returns:
        np.ndarray: 2D autocorrelation function. 
            Length will be 2 * oversample * output_range
        np.ndarray: 1D radial sum (None for radial=False). 
            Length will be oversample * output_ramge
    """
    
    from numpy import fft
    from itertools import repeat
    
    # if a stack of pattern data is supplied, call recursively on single shots
    if isinstance(npk, np.ndarray) and len(npk) > 1:
        _all_args = zip(npk, x, y, repeat(None) if I is None else I)
        _kwargs = {'roi_length': roi_length, 
                   'output_radius': output_radius,
                   'oversample': oversample,
                   'radial': radial,
                   'px_ang': px_ang}
        if execution == 'single-threaded':
            res = [get_acf(*_args, **_kwargs) for _args in _all_args]
        else:
            with (ProcessPoolExecutor() if execution=='processes' 
              else ThreadPoolExecutor()) as exc:
                ftrs = [exc.submit(get_acf, *_args, **_kwargs) for _args in _all_args]
                wait(ftrs, return_when='FIRST_EXCEPTION');
                # for ftr in ftrs:
                #     if ftr.exception() is not None:
                #         raise ftr.exception()
                res = [f.result() for f in ftrs]
        return (np.stack(stk) for stk in zip(*res))  
    
    
    sz = roi_length * oversample
    rng = output_radius * oversample
    
    pkx = (oversample * x[:npk]).round().astype(int) + sz//2
    pky = (oversample * y[:npk]).round().astype(int) + sz//2
    pkI = None if I is None else I[:npk]
    
    if px_ang is not None:
        t_par = (pkx**2 + pky**2)**.5 * px_ang
        acorr = 2*np.sin(np.arctan(t_par)/2) / t_par
    else:
        acorr = 1
        
    valid = (pkx >= 0) & (pkx < sz) & (pky >= 0) & (pky < sz)
    pkx, pky, pkI = acorr*pkx[valid], acorr*pky[valid], 1 if I is None else pkI[valid]
    dense = np.zeros((sz, sz), dtype=np.float if I is None else np.uint8)
    dense[pkx, pky] = pkI if I is not None else 1
    acf = fft.ifft2(np.abs(fft.fft2(dense))**2)
    acf = fft.ifftshift(acf).real
    if I is None:
        # if no intensities were given, the result is (should be) 
        # integer, up to numerical noise
        acf = acf.round().astype(np.uint8)
        if acf[sz//2, sz//2] != sum(valid):
            warn(f'Autocorrelation center pixel ({acf[sz//2, sz//2]}) does not equal the peak number ({sum(valid)})!')
    acf[sz//2, sz//2] = 0 # remove self-correlation (which will be equal to the peak number)
    if radial:
        rad = proc2d.radial_proj(acf, min_size=rng, max_size=rng, 
                             my_func=np.sum, x0=sz//2, y0=sz//2)
    else:
        rad = None
    return acf[sz//2-rng:sz//2+rng, sz//2-rng:sz//2+rng], rad


def get_pk_data(n_pk: np.ndarray, pk_x: np.ndarray, pk_y: np.ndarray, 
                ctr_x: np.ndarray, ctr_y: np.ndarray, pk_I: Optional[np.ndarray] = None,
                opts: Optional[PreProcOpts] = None,
                peakmask=None, return_vec=True, pxs=None, 
                clen=None, wl=None, el_rat=None, el_ang=None):
    
    if peakmask is None:
        peakmask = np.ones_like(pk_x, dtype=np.float)
        for N, row in zip(n_pk, peakmask):
            row[N:] = np.nan
       
    if opts is not None:
        pxs = opts.pixel_size if pxs is None else pxs
        clen = opts.cam_length if clen is None else clen
        wl = opts.wavelength if wl is None else wl
        el_rat = opts.ellipse_ratio if el_rat is None else el_rat
        el_ang = opts.ellipse_angle if el_ang is None else el_ang
        
    #     assert (np.nansum(peakmask, axis=1) == n_pk).all()      
    pk_xr, pk_yr = pk_x - ctr_x.reshape(-1,1), pk_y - ctr_y.reshape(-1,1)
    pk_xr, pk_yr = pk_xr * peakmask, pk_yr * peakmask
    
    # ellipticity correction
    if el_rat is not None and (el_rat != 1):
        c, s = np.cos(np.pi/180*el_ang), np.sin(np.pi/180*el_ang)
        pk_xrc, pk_yrc = 1/el_rat**.5*(c*pk_xr - s*pk_yr), el_rat**.5*(s*pk_xr + c*pk_yr)
        pk_xrc, pk_yrc = c*pk_xrc + s*pk_yrc, - s*pk_xrc + c*pk_yrc
    else:
        pk_xrc, pk_yrc = pk_xr, pk_yr
    
    res = {'peakXPosRaw': pk_x,   'peakYPosRaw': pk_y, 
           'peakXPosRel': pk_xr,  'peakYPosRel': pk_yr,
           'peakXPosCor': pk_xrc, 'peakYPosCor': pk_yrc,
           'nPeaks': n_pk}
    
    if pk_I is not None:
        res['peakTotalIntensity'] = pk_I

    if return_vec:
        if (pxs is None) or (clen is None) or (wl is None):
            raise ValueError('Cannot return angle parameters without pxs, clen, wl.')   
        pk_r = (pk_xrc**2 + pk_yrc**2)**.5        
        pk_tt = np.arctan(pxs * pk_r / clen)
        pk_az = np.arctan2(pk_yrc, pk_xrc)
        pk_d = wl/(2*np.sin(pk_tt/2))
        res.update({'peakTwoTheta': pk_tt, 'peakAzimuth': pk_az, 'peakD': pk_d})
    
    return res
    
class Cell(object):
    """
    Partially taken from the PyFAI package, with some simplifications 
    and speed enhancements for d-spacing calculation, as well as a 
    new refinement function.
    
    Calculates d-spacings and cell volume as described in:
    http://geoweb3.princeton.edu/research/MineralPhy/xtalgeometry.pdf
    """
    lattices = ["cubic", "tetragonal", "hexagonal", "rhombohedral", 
                "orthorhombic", "monoclinic", "triclinic"]
    ctr_types = {"P": "Primitive",
             "I": "Body centered",
             "F": "Face centered",
             "C": "Side centered",
             "R": "Rhombohedral"}

    def __init__(self, a=1, b=1, c=1, alpha=90, beta=90, gamma=90, 
                 lattice_type="triclinic", centering="P", 
                 unique_axis="c", d_min=2):
        """Constructor of the Cell class:

        Crystallographic units are Angstrom for distances and degrees for angles

        :param a,b,c: unit cell length in Angstrom
        :param alpha, beta, gamma: unit cell angle in degrees
        :param lattice: "cubic", "tetragonal", "hexagonal", "rhombohedral", "orthorhombic", "monoclinic", "triclinic"
        :param lattice_type: P, I, F, C or R
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lattice_type = lattice_type if lattice_type in self.lattices else "triclinic"
        self._centering = centering
        self.unique_axis = unique_axis
        self._volume = None
        self.selection_rules = []
        "contains a list of functions returning True(allowed)/False(forbidden)/None(unknown)"
        self.centering = "P"
        self.hkl = None
        self._d_min = d_min
        self.init_hkl(d_min)
 
    def __repr__(self, *args, **kwargs):
        return "%s %s cell (unique %s) a=%.4f b=%.4f c=%.4f alpha=%.3f beta=%.3f gamma=%.3f" % \
            (self.ctr_types[self.centering], self.lattice_type, self.unique_axis,
             self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    @classmethod
    def cubic(cls, a, centering="P"):
        """Factory for cubic lattice_types

        :param a: unit cell length
        """
        a = float(a)
        self = cls(a, a, a, 90, 90, 90,
                   lattice_type="cubic", centering=centering)
        return self

    @classmethod
    def tetragonal(cls, a, c, centering="P"):
        """Factory for tetragonal lattice_types

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(a, a, float(c), 90, 90, 90,
                   lattice_type="tetragonal", centering=centering)
        return self

    @classmethod
    def orthorhombic(cls, a, b, c, centering="P"):
        """Factory for orthorhombic lattice_types

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        """
        self = cls(float(a), float(b), float(c), 90, 90, 90,
                   lattice_type="orthorhombic", centering=centering)
        return self

    @classmethod
    def hexagonal(cls, a, c, centering="P"):
        """Factory for hexagonal lattice_types

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(a, a, float(c), 90, 90, 120,
                   lattice_type="hexagonal", centering=centering)
        return self

    @classmethod
    def monoclinic(cls, a, b, c, beta, centering="P"):
        """Factory for hexagonal lattice_types

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        :param beta: unit cell angle
        """
        self = cls(float(a), float(b), float(c), 90, float(beta), 90,
                   centering=centering, lattice_type="monoclinic", 
                   unique_axis='b')
        return self

    @classmethod
    def rhombohedral(cls, a, alpha, centering="P"):
        """Factory for hexagonal lattice_types

        :param a: unit cell length
        :param alpha: unit cell angle
        """
        a = float(a)
        alpha = float(a)
        self = cls(a, a, a, alpha, alpha, alpha,
                   lattice_type="rhombohedral", centering=centering)
        return self

    @classmethod
    def diamond(cls, a):
        """Factory for Diamond type FCC like Si and Ge

        :param a: unit cell length
        """
        self = cls.cubic(a, centering="F")
        self.selection_rules.append(lambda h, k, l: not((h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0) and ((h + k + l) % 4 != 0)))
        return self
       
    @property
    def volume(self):
        if self._volume is None:
            self._volume = self.a * self.b * self.c
            if self.lattice_type not in ["cubic", "tetragonal", "orthorhombic"]:
                cosa = np.cos(self.alpha * np.pi / 180.)
                cosb = np.cos(self.beta * np.pi / 180.)
                cosg = np.cos(self.gamma * np.pi / 180.)
                self._volume *= np.sqrt(1 - cosa ** 2 - cosb ** 2 - cosg ** 2 
                                        + 2 * cosa * cosb * cosg)
        return self._volume

    @property
    def centering(self):
        return self._centering

    @centering.setter
    def centering(self, centering):
        self._centering = centering if centering in self.ctr_types else "P"
        self.selection_rules = [lambda h, k, l: ~((h == 0) & (k == 0) & (l == 0))]
        if self._centering == "I":
            self.selection_rules.append(lambda h, k, l: (h + k + l) % 2 == 0)
        if self._centering == "F":
            self.selection_rules.append(lambda h, k, l: np.isin(h % 2 + k % 2 + l % 2, (0, 3)))
        if self._centering == "R":
            self.selection_rules.append(lambda h, k, l: ((h - k + l) % 3 == 0))
        
    def init_hkl(self, d_min: float = 5.):
        """Sets up a grid with valid Miller indices for this lattice.
        Useful to pre-compute the indices before running any optimization,
        which speeds up the computation.

        Args:
            d_min (float, optional): Minimum d-spacing, in A. Defaults to 5.
        """
        hmax = int(np.ceil(self.a / d_min))
        kmax = int(np.ceil(self.b / d_min))
        lmax = int(np.ceil(self.c / d_min))
        hkl = np.mgrid[-hmax:hmax+1, -kmax:kmax+1, -lmax:lmax+1]
        valid = np.stack([r(*hkl) for r in self.selection_rules], axis=0).all(axis=0)
        self.hkl = tuple(H[valid].ravel() for H in hkl)
        d = self.d(d_min=None)
        self.hkl = tuple(H[d >= d_min] for H in self.hkl)
        self._d_min = d_min
        
    def d(self, d_min=None, unique=False, a=None, b=None, c=None, 
          alpha=None, beta=None, gamma=None):
        """Calculates d-spacings for the cell. Cell parameters can
        transiently be changed, which does *not* affect the values
        stored with the cell object. This is useful in the context
        of optimization.

        Args:
            d_min (float, optional): Minimum d-spacing. If None, uses
                the stored value of the object which can be set using 
                init_hkl. Leaving it at None significantly speeds
                up the computation, which is recommended for
                refinements. Defaults to None.
            unique (bool, optional): if True, only unique d-spacings
                are returned. Otherwise, all spacings are returned which
                are ordered the same way as in the object's hkl attribute. 
                Defaults to False.
            a (float, optional): Temporary cell length. Defaults to None.
            b (float, optional): Temporary cell length. Defaults to None.
            c (float, optional): Temporary cell length. Defaults to None.
            alpha (float, optional): Temporary cell angle. Defaults to None.
            beta  (float, optional): Temporary cell angle. Defaults to None.
            gamma (float, optional): Temporary cell angle. Defaults to None.

        Returns:
            np.array: Array of d-spacings
        """

        
        a = self.a if a is None else a
        b = self.b if b is None else b
        c = self.c if c is None else c
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        
        if (d_min is not None) and (d_min != self._d_min):
            self.init_hkl(d_min)
        
        h, k, l = self.hkl
        
        if self.lattice_type in ["cubic", "tetragonal", "orthorhombic"]:
            invd2 = (h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2
        else:
            cosa, sina = np.cos(alpha * np.pi / 180), np.sin(alpha * np.pi / 180)
            cosb, sinb = np.cos(beta * np.pi / 180), np.sin(beta * np.pi / 180)
            cosg, sing = np.cos(gamma * np.pi / 180), np.sin(gamma * np.pi / 180)
            S11 = (b * c * sina) ** 2
            S22 = (a * c * sinb) ** 2
            S33 = (a * b * sing) ** 2
            S12 = a * b * c * c * (cosa * cosb - cosg)
            S23 = a * a * b * c * (cosb * cosg - cosa)
            S13 = a * b * b * c * (cosg * cosa - cosb)

            invd2 = (S11 * h * h +
                     S22 * k * k +
                     S33 * l * l +
                     2 * S12 * h * k +
                     2 * S23 * k * l +
                     2 * S13 * h * l)
            invd2 /= (self.volume) ** 2
            
        return np.sqrt(1 / (np.unique(invd2) if unique else invd2)) 
    
    d_spacing = d  
    
    def export(self, filename='refined.cell'):
        from textwrap import dedent
        """Exports the cell to a CrystFEL cell file.

        Args:
            filename (str, optional): Cell file name. Defaults to 'refined.cell'.
        """
        
        cellfile = dedent(f'''
        CrystFEL unit cell file version 1.0
        
        lattice_type = {self.lattice_type}
        centering = {self.centering}
        unique_axis = {self.unique_axis}
        
        a = {self.a:.3f} A
        b = {self.b:.3f} A
        c = {self.c:.3f} A
        
        al = {self.alpha:.2f} deg
        be = {self.beta:.2f} deg
        ga = {self.gamma:.2f} deg
        ''').strip()
        
        with open(filename, 'w') as fh:
            fh.write(cellfile)
            
    def refine_powder(self, svec, pattern, method='distance',
                  fill=0.1, min_prom=0., min_height=0., 
                  weights='prom', length_bound=2., angle_bound=3.,
                  **kwargs):
        """Refine unit cell parameters against a powder pattern.
        The refinement is done using a least-squares fit, where you can
        pick three different cost functions:
        
        * 'distance': the positions of the peaks in the powder pattern
            are detected. For each peak, the distance to the closest
            d-spacing is computed.
        * 'xcorr': the inverse values of the powder pattern at the
            d-spacings are computed.
        * 'derivative': the derivative of the powder pattern at the
            d-spacings are computed
            
        Depending on the chosen method, further parameters can be set.
        The function returns a new Cell object with refined parameters, and
        a structure with some useful information.

        Args:
            svec (np.ndarray): scattering vector (x-axis) of the powder pattern,
                expressed in inverse nanometer (not angstrom - following
                CrystFEL convention).
            pattern (np.ndarray): powder pattern at values svec (y-axis)
            method (str, optional): Cost function. See description. 
                Defaults to 'distance'.
            fill (float, optional): Fill value for out-of-range or zero-count
                s-vectors if method is 'derivative' or 'xcorr'. Defaults to 0.1.
            min_prom (float, optional): Minimum prominence of peaks (that is,
                height relative to its vicinity) if method is 'distance'. Increase
                if too many small peaks are spuriously detected. Usually it is
                a good idea. Defaults to 0.
            min_height (float, optional): Minimum peak height to be detected. 
                Usually min_prom is the better parameter. Defaults to 0.
            weights (str, optional): Weights of the peaks for the least-squares
                optimization if method is 'derivative'. Can be 'prom' or 'height'.
                Defaults to 'prom'.
            length_bound (float, optional): Bound range for cell lengths, in A. 
                Defaults to 2.
            angle_bound (float, optional): Bound range for cell angles. Defaults to 3.
            **kwargs: Further arguments will be passed on to scipy.least_sqaures

        Returns:
            tuple: 2-Tuple of a new Cell object with the refined parameters, and
                a structure with useful information from the optimization, including
                the peak positions and heights if method was 'distance'.
        """
        
        from scipy.interpolate import interp1d
        from scipy.optimize import least_squares
        
        # find out which parameters should be optimized
        if self.lattice_type == 'triclinic':
            parameters = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        elif self.lattice_type == 'monolinic':
            parameters = ['a', 'b', 'c', 'beta']
        elif self.lattice_type == 'orthorhombic':
            parameters = ['a', 'b', 'c']
        elif self.lattice_type == 'tetragonal':
            parameters = ['a', 'c']
        elif self.lattice_type == 'cubic':
            parameters = ['a']
        elif self.lattice_type == 'hexagonal':
            parameters = ['a', 'c']
        elif self.lattice_type == 'rhombohedral':
            parameters = ['a', 'alpha']
        else:
            raise Exception('This should not happen. Yell at Robert.')
        
        _, unique_pos = np.unique(self.d(), return_index=True) # unique d-spacings
        p0 = [getattr(self, cpar) for cpar in parameters]
        dsp = lambda p: self.d(**{cpar: p[ii] for ii, cpar in enumerate(parameters)})[unique_pos]
        bounds = ([getattr(self, cpar) - (length_bound if cpar in 'abc' else angle_bound) 
                  for cpar in parameters],
                  [getattr(self, cpar) + (length_bound if cpar in 'abc' else angle_bound) 
                  for cpar in parameters])

        if method == 'xcorr':
            cost_profile = interp1d(svec, 1/np.where(pattern != 0, pattern, fill), 
                                    bounds_error=False, fill_value=fill)
            cost = lambda p: cost_profile(10/dsp(p))
            pk_pos = pk_height = pk_prom = []

        elif method == 'derivative':
            cost_profile = interp1d(svec[1:]/2+svec[:-1]/2, np.diff(pattern),
                                    bounds_error=False, fill_value=0)
            cost = lambda p: cost_profile(10/dsp(p))
            pk_pos = pk_height = pk_prom = []

        elif method == 'distance':
            from scipy.signal import find_peaks        
            lim = 0.95 * 10/dsp(p0).min()
            pkdat = find_peaks(pattern[svec < lim], height=min_height, prominence=min_prom)
            pk_pos = svec[pkdat[0]]
            pk_height = pkdat[1]['peak_heights']
            pk_prom = pkdat[1]['prominences']
            w = pk_prom if weights == 'prom' else 1
            w = pk_height if weights == 'height' else 1
            cost = lambda p: 100*w * (np.abs(10/dsp(p).reshape(1,-1) 
                                             - pk_pos.reshape(-1,1))).min(axis=1)
            
        else:
            raise ValueError(f'Unknown refinement method {method}')

        cost_init = 0.5 * (cost(p0)**2).sum()
        lsq = least_squares(cost, p0, bounds=bounds, **kwargs)
        
        # return a new cell with the optimized parameters
        C_ref = getattr(self, self.lattice_type)(centering=self._centering,
            **{cpar: lsq.x[ii] for ii, cpar in enumerate(parameters)})
        C_ref.selection_rulse = self.selection_rules
        C_ref.init_hkl(self._d_min)
        
        info = {'lsq_result': lsq,
                'initial_cost': cost_init}
        if method == 'distance':
            info.update({'peak_position': pk_pos,
                         'peak_height': pk_height,
                         'peak_prominence': pk_prom})
            
        if not lsq.success:
            warn('Powder refinement did not converge!')
        
        return C_ref, info
