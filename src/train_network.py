#!/usr/bin/env python

""" Some utilities for Face Recognition Algorithm """

__author__ = "Venkatraman Narayanan"
__version__ = "1.0"
__email__ = "vnarayan@terpmail.umd.edu"


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, LocallyConnected2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import create_subs

train_data, cv_data, test_data = create_subs()

X_train, Y_train = (np.asarray(train_data[0]), np.asarray(train_data[1]))

X_train = X_train / 255


def FRModel(input_shape, classes):
    
    X_input = Input(input_shape)
    
    X = Conv2D(32, (11, 11), strides = (1, 1), name = 'C1')(X_input)
    
    X = MaxPooling2D((3, 3), strides = (2, 2), name='M2')(X)
    
    X = Conv2D(16, (9, 9), strides = (1, 1), name = 'C3')(X)
    
    X = LocallyConnected2D(16, (9, 9), name = 'L4')(X)
    
    X = LocallyConnected2D(16, (7, 7), name = 'L5')(X)
    
    X = LocallyConnected2D(16, (5, 5), name = 'L6')(X)
    
    X = Flatten()(X)
    
    X = Dense(4096, activation='relu', name='F7')(X)
    
    X = Dense(classes, activation='softmax', name='F8')(X)
    
    model = Model(inputs = X_input, outputs = X, name='FRModel')
    
    return model


    
    
    
    






