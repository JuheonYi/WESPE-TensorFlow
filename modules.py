from __future__ import division
import time
import tensorflow as tf
#import tensorlayer as tl
import scipy.misc
import scipy.io
import numpy as np
from utils import *
from ops import *

def resblock(feature_in, num):
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(feature_in, 64, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    #temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_1' %num))
    temp = tf.nn.relu(temp)
        
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    #temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_2' %num))
    temp = tf.nn.relu(temp)
    return temp + feature_in

def generator_network(image, var_scope = 'generator'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # conv. layer before residual blocks 
        b1_in = tf.layers.conv2d(image, 64, 9, strides = 1, padding = 'SAME', name = 'CONV_1', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        b1_in = tf.nn.relu(b1_in)
            
        # residual blocks
        b1_out = resblock(b1_in, 1)
        b2_out = resblock(b1_out, 2)
        b3_out = resblock(b2_out, 3)
        b4_out = resblock(b3_out, 4)
            
        # conv. layers after residual blocks
        temp = tf.layers.conv2d(b4_out, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_2', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 64, 3, strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.nn.relu(temp)
        temp = tf.layers.conv2d(temp, 3, 1, strides = 1, padding = 'SAME', name = 'CONV_5', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        return temp

def discriminator_network(image, var_scope = 'discriminator', preprocess = 'gray'):
    with tf.variable_scope(var_scope, reuse = tf.AUTO_REUSE):
        if preprocess == 'gray':
            #convert to grayscale image
            print("Discriminator-texture")
            image_processed = tf.image.rgb_to_grayscale(image)
        elif preprocess == 'blur':
            print("Discriminator-color (blur)")
            image_processed = gaussian_blur(image)
        else:
            print("Discriminator-color (none)")
            image_processed = image
            
        # conv layer 1 
        temp = tf.layers.conv2d(image_processed, 48, 11, strides = 4, padding = 'SAME', name = 'CONV_1', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = lrelu(temp)
            
        # conv layer 2
        temp = tf.layers.conv2d(temp, 128, 5, strides = 2, padding = 'SAME', name = 'CONV_2', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_2')
        temp = lrelu(temp)
            
        # conv layer 3
        temp = tf.layers.conv2d(temp, 192, 3, strides = 1, padding = 'SAME', name = 'CONV_3', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_3')
        temp = lrelu(temp)
            
        # conv layer 4
        temp = tf.layers.conv2d(temp, 192, 3, strides = 1, padding = 'SAME', name = 'CONV_4', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_4')
        temp = lrelu(temp)
            
        # conv layer 5
        temp = tf.layers.conv2d(temp, 128, 3, strides = 2, padding = 'SAME', name = 'CONV_5', kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
        temp = tf.layers.batch_normalization(temp, name = 'BN_5')
        temp = lrelu(temp)
            
        # FC layer 1
        fc_in = tf.contrib.layers.flatten(temp)
        fc_out = tf.layers.dense(fc_in, units = 1024, activation = None)
        fc_out = lrelu(fc_out)
            
        # FC layer 2
        logits = tf.layers.dense(fc_out, units = 1, activation = None)
        probability = tf.nn.sigmoid(logits)
    return logits, probability