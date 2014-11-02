import pylab as pl
import cv2
import numpy as np
import os
rand = np.random
from scipy.io import savemat,loadmat
from scipy.ndimage.filters import gaussian_filter
try:
    from tifffile import imread as timread
except:
    pass

deconv_path = 'path_to_matlab_output'
sim_path = 'path_to_sim'

deconv = loadmat(deconv_path)
sim = np.load(sim_path)

