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
    
def load_dataset(config):
    phone_list = sorted(glob(config.train_path_phone))
    DIV2K_list = sorted(glob(config.train_path_DIV2K))
    print("Dataset: %s, %d images" %(config.dataset_name, len(phone_list)))
    print("DIV2K: %d images" %(len(DIV2K_list)))
    start_time = time.time()
    dataset_phone = [scipy.misc.imread(filename, mode = "RGB") for filename in phone_list]
    dataset_DIV2K = [scipy.misc.imread(filename, mode = "RGB") for filename in DIV2K_list]
    print("%d images loaded! setting took: %4.4fs" % (len(dataset_phone), time.time() - start_time))
    return dataset_phone, dataset_DIV2K

def get_batch(dataset_phone, dataset_DIV2K, config, start = 0):
    phone_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')
    DIV2K_batch = np.zeros([config.batch_size, config.patch_size, config.patch_size, config.channels], dtype = 'float32')

    for i in range(config.batch_size):
        index = np.random.randint(len(dataset_phone))
        phone_patch = dataset_phone[index]
        index = np.random.randint(len(dataset_DIV2K))
        DIV2K_img = dataset_DIV2K[index]
        #print("img shape:", DIV2K_img.shape)
        
        patch_size = config.patch_size
        H = np.random.randint(DIV2K_img.shape[0]-patch_size)
        W = np.random.randint(DIV2K_img.shape[1]-patch_size)
        DIV2K_patch = DIV2K_img[H: H+patch_size, W: W+patch_size, :]
        #print("patch shape:", DIV2K_patch.shape)
        
        #imageio.imwrite("./DIV2Kimg.png", DIV2K_img)
        #imageio.imwrite("./DIV2Kpatch.png", DIV2K_patch)

        # randomly flip, rotate patch (assuming that the patch shape is square)
        if config.augmentation == True:
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 0)
                DIV2K_patch = np.flip(DIV2K_patch, axis = 0)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.flip(phone_patch, axis = 1)
                DIV2K_patch = np.flip(DIV2K_patch, axis = 1)
            prob = np.random.rand()
            if prob > 0.5:
                phone_patch = np.rot90(phone_patch)
                DIV2K_patch = np.rot90(DIV2K_patch)

        phone_batch[i,:,:,:] = preprocess(phone_patch) # pre/post processing function is defined in ops.py
        DIV2K_batch[i,:,:,:] = preprocess(DIV2K_patch)
    return phone_batch, DIV2K_batch