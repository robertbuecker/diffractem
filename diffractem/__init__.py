import numpy as np

__all__ = ['compute', 'io', 'proc2d', 'tools', 'map_image.py', 'models']


def gap_pixels():
    """Returns the gap pixels of the Lambda detector as binary mask"""
    gaps = np.zeros((516, 1556), dtype=np.bool)
    for k in range(255, 1296, 260):
        gaps[:, k:k+6] = True
    gaps[255:261] = True
    return gaps


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
