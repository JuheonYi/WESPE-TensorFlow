from __future__ import division
import numpy as np
import os
import os.path
import time
from glob import glob
import scipy.misc
import scipy.io
from scipy.misc import imresize

from random import shuffle
import imageio
import cv2
import math
from ops import *

def load_img(image_path, mode = "RGB"):
    if mode == "RGB":
        return scipy.misc.imread(image_path, mode = "RGB")
    else: 
        return scipy.misc.imread(image_path, mode = "YCbCr")

def save_img(img, path):
    imageio.imwrite(path, img)
    return 0

def resize_img(x, shape):
    x = np.copy(x).astype(np.uint8)
    y = imresize(x, shape, interp='bicubic')
    return y

def calc_PSNR(img1, img2):
    #assume RGB image
    target_data = np.array(img1, dtype=np.float64)
    ref_data = np.array(img2,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    if rmse == 0:
        return 100
    else:
        return 20*math.log10(255.0/rmse)