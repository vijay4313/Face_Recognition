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
from keras.utils import to_categorical
from keras.models import model_from_json

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import create_subs


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
    
    model = Model(inputs = X_input, outputs = X,  name='FRModel')
    
    return model


def data_process():
    
    train_data, cv_data, test_data, Y_str2int = create_subs()
    
    X_train, Y_train = (np.asarray(train_data[0]), np.asarray(train_data[1]))
    
    X_val, Y_val = (np.asarray(cv_data[0]), to_categorical(np.asarray(cv_data[1])))
    
    X_test, Y_test = (np.asarray(test_data[0]), to_categorical(np.asarray(test_data[1])))
    
    num_classes = np.maximum(np.asarray(Y_str2int.values()))
    
    X_train = X_train / 255
    
    X_val = X_val / 255
    
    X_test = X_test / 255
    
    Y_train_one_hot = to_categorical(Y_train, num_classes)
    
    Y_val_one_hot = to_categorical(Y_val, num_classes)
    
    Y_test_one_hot = to_categorical(Y_test, num_classes)
    
    tr_data = (X_train, Y_train_one_hot)
    
    val_data = (X_val, Y_val_one_hot)
    
    tst_data = (X_test, Y_test_one_hot)
    
    return (tr_data, val_data, tst_data, num_classes)
    
    


def train():
    
    train, val, test, num_classes = data_process()
    
    X_train, Y_train = train
    
    X_test, Y_test = test
    
    mdl = FRModel(X_train.shape, num_classes)
    
    mdl.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    mdl.fit(X = X_train, Y = Y_train, epochs = 100, batch_size = 64, validation_data = val)
    
    test_scores = mdl.evaluate(X_test, Y_test)
    
    print("%s: %.2f%%" % (mdl.metrics_names[1], test_scores[1]*100))
    
    # serialize model to JSON
    model_json = mdl.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mdl.save_weights("model.h5")
    print("Saved model to disk")


def img_encoding(img_X):
    
    X = img_X / 255
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    
    
    siam_layer = 'F7'
    
    siam_model = Model(inputs = loaded_model.input, outputs = loaded_model.get_layer(siam_layer).output)
    
    siam_encoding = siam_model.predict(X)
    
    return siam_encoding
    
    
    
    


    
    
    
    






