import pylab as pl
import cv2
import numpy as np
import os
rand = np.random
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
try:
    from tifffile import imread as timread
except:
    pass

movie = 'output/test_mov'

sim = np.load(movie+'.npz')

