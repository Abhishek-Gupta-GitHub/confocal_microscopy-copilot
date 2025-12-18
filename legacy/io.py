import tifffile as tiff
import numpy as np

def load_tiff_stack(path):
    data = tiff.imread(path)
    return np.asarray(data)
