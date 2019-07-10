# Friedel-pair refinement
from scipy.optimize import least_squares, leastsq
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION


def _ctr_from_pks(pkl: np.ndarray, p0: np.ndarray,
                  int_weight: bool = False, sigma: float = 2.0, bound: float = 5.0, label: str = None):
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
        return lsq.x, 1 / lsq.cost, label


def center_friedel(peaks, shots=None, p0=(778, 308), colnames=('fs/px', 'ss/px'), sigma=2,
                   minpeaks=4, maxres=150):
    colnames = list(colnames)
    p0 = np.array(p0)

    with ProcessPoolExecutor() as p:
        futures = []
        for grp, pks in peaks.groupby(['file', 'Event']):
            pkl = pks.loc[:, colnames].values
            rsq = (pkl[:, 0] - p0[0]) ** 2 + (pkl[:, 1] - p0[1]) ** 2
            if maxres is not None:
                pkl = pkl[rsq < maxres ** 2, :]
            if (minpeaks is None) or pkl.shape[0] > minpeaks:
                futures.append(p.submit(_ctr_from_pks, pkl, p0, sigma=sigma, label=grp))

    wait(futures, return_when=FIRST_EXCEPTION)

    # reformat result into a dataframe
    cpos = pd.concat([pd.DataFrame(data=np.array([t.result()[2] for t in futures]), columns=['file', 'Event']),
                      pd.DataFrame(data=np.array([t.result()[0] for t in futures]), columns=['beam_x', 'beam_y'])],
                     axis=1)

    if shots is not None:
        # include shots that were not present in the peaks table
        cpos = shots[['file', 'Event']].merge(cpos, on=['file', 'Event'], how='left'). \
            fillna({'beam_x': p0[0], 'beam_y': p0[1]})

    return cpos
