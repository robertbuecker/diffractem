import numpy as np

__all__ = ['compute', 'io', 'proc2d', 'tools', 'map_image', 'pre_proc_opts']

def version():
    try:
        with open(__file__.rsplit('/',1)[0] + '/../version.txt') as fh:
            return fh.readline().strip()
    except FileNotFoundError:
        return 'Could not determine diffractem version'


def gap_pixels(detector='Lambda750k'):
    """Returns the gap pixels of the Lambda detector as binary mask"""
    if detector == 'Lambda750k':
        gaps = np.zeros((516, 1556), dtype=np.bool)
        for k in range(255, 1296, 260):
            gaps[:, k:k+6] = True
        gaps[255:261] = True
    else:
        raise ValueError(f'Unknown detector: {detector}')
    return gaps


def panel_pix(panel_id=1, pxmask=None, img=None, 
              detector='Lambda750k', include_gap=True):
    
    if detector == 'Lambda750k':
        shape = (1556, 516)
        panel_size = 256 if include_gap else 255
        panel_gap = 4 if include_gap else 6
        cutoff = (60, 0)
        row, col = divmod(panel_id-1, 6)
        if panel_id > 6:
            col = 5-col        
        if panel_id > 12:
            raise ValueError('panel_id cannot be larger than 12')
    else:
        raise ValueError(f'Unknown detector {detector}')
    
    mask = np.zeros((shape[1], shape[0]))
    #print(row,col)
    cstart = col*(panel_size + panel_gap)
    rstart = row*(panel_size + panel_gap)
    mask[rstart:rstart+panel_size, cstart:cstart+panel_size] = 1
    mask[:(cutoff[1]+1), :(cutoff[0]+1)] = 0
    mask[-(cutoff[1]+1):, -(cutoff[0]+1):] = 0
    if pxmask is not None:
        mask = mask - pxmask
    if img is None:
        return mask == 1
    else:
        cimg = img[rstart:rstart+panel_size, cstart:cstart+panel_size]
        if pxmask is not None:
            pm = pxmask[rstart:rstart+panel_size, cstart:cstart+panel_size]
        else:
            pm = np.zeros_like(cimg)
        cimg[pm != 0] = -1
        return cimg


def normalize_names(strin):
    strout = strin
    for character in [' ', '/', '(', ')', '-']:
        strout = strout.replace(character, '_')
    return strout


def normalize_keys(dictionary):
    d = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            d[normalize_names(k)] = normalize_keys(v)
        else:
            d[normalize_names(k)] = v
    return d
