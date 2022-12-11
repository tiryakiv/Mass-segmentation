# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:05:38 2021

@author:  Asma Baccouche
"""


from __future__ import print_function

import os, glob
from skimage.io import imsave
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, concatenate, Conv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, UpSampling2D, Add, Activation
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.layers import add,multiply
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
#import matplotlib.pyplot as plt
#from data import load_train_data, load_test_data
#from sklearn.model_selection import train_test_split
#K.set_image_data_format('channels_last')  # TF dimension ordering in this code


img_rows = 640
img_cols = 640
smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#def iou_coef(y_true, y_pred, smooth=1):
#      intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#      union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#      iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#      return iou

def sens(y_true, y_pred):
    num=K.sum(K.multiply(y_true, y_pred))
    denom=K.sum(y_true)
    if denom==0:
        return 1
    else:
        return  num/denom

def spec(y_true, y_pred):
    num=K.sum(K.multiply(y_true==0, y_pred==0))
    denom=K.sum(y_true==0)
    if denom==0:
        return 1
    else:
        return  num/denom

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))


def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    BCE = K.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(0.8 * K.pow((1-BCE_EXP), 2.) * BCE)
    return focal_loss

def seg_loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))

def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):
	shape_x = K.int_shape(x)  # 32
	shape_g = K.int_shape(g)  # 16
	theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
	shape_theta_x = K.int_shape(theta_x)
	phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
	upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
	concat_xg = add([upsample_g, theta_x])
	act_xg = Activation('relu')(concat_xg)
	psi = Conv2D(1, (1, 1), padding='same')(act_xg)
	sigmoid_xg = Activation('sigmoid')(psi)
	shape_sigmoid = K.int_shape(sigmoid_xg)
	upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
	upsample_psi = expend_as(upsample_psi, shape_x[3])
	y = multiply([upsample_psi, x])
	result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
	result_bn = BatchNormalization()(result) 
	return result_bn

def UnetGatingSignal(inputs, is_batchnorm=False):
    shape = K.int_shape(inputs)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(inputs)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def get_aunet():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    gating = UnetGatingSignal(conv5, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 256)
    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv5), attn_1], axis=3)  
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    gating = UnetGatingSignal(conv6, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 128)
    up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv6), attn_2], axis=3) 
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    gating = UnetGatingSignal(conv7, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 64)
    up8 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv7), attn_3], axis=3) 
    
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    gating = UnetGatingSignal(conv8, is_batchnorm=True)  
    attn_4 = AttnGatingBlock(conv1, gating, 32)
    up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv8), attn_4], axis=3) 
    
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    #model.compile(optimizer=Adam(1e-4), loss=[seg_loss], metrics=[dice_coef, iou_coef])
    return model