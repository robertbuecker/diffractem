# Friedel-pair refinement
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import current_process
from typing import Optional


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
                         / pkl.shape[0]

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
        peaks {[pd.DataFrame]} -- [peaks list for entire data set, as returned by StreamParser]
    
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
