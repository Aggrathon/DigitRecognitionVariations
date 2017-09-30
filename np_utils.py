"""
    This script contains some useful non tensorflow functions
"""
import numpy as np

def np_onehot(size: int, num: int, type=float):
    """
        Returns a onehot numpy vector
    """
    vec = np.zeros(size, type)
    vec[num] = 1
    return vec
