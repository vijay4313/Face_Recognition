#!/usr/bin/env python

""" Some utilities for Face Recognition Algorithm """

__author__ = "Venkatraman Narayanan"
__version__ = "1.0"
__email__ = "vnarayan@terpmail.umd.edu"

import os
import cv2
import random
import sys


super_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

front_path = os.path.join(super_dir_path, 'face-frontalization')
sys.path.append(front_path)

import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import check_resources as check
import numpy as np


def preprocess(img):
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()
    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(front_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    
    lmarks = feature_detection.get_landmarks(img)
    
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    
    eyemask = np.asarray(io.loadmat(front_path + '/frontalization_models/eyemask.mat')['eyemask'])
    
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    
    return frontal_sym


def load_data():
    X = []
    Y = []
    path = os.path.join(super_dir_path, 'data')
    for root, dirs, files in os.walk(path):
        for name in files:
            
#            if (len(Y) > 10):
#                return (X, Y)
#                break
            
            if name.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, name))
                X.append(preprocess(img))
                Y.append(root.split('/')[-1])
    return (X, Y)


def create_subs(seed = 1):
    random.seed(seed)
    
    X, Y = load_data()
    Y_str2int = dict([(y, x+1) for x, y in enumerate(sorted(set(Y)))])
    Y_encoded = [Y_str2int[y] for y in Y]
    tr_samples = set(random.sample(range(len(X)), int(0.6 * len(X))))
    rem_samples = set(range(len(X))).difference(tr_samples)
    cv_samples = set(random.sample(rem_samples, int(0.2 * len(X))))
    test_samples = rem_samples.difference(cv_samples)
    
    tr_data = ([X[i] for i in tr_samples], [Y_encoded[i] for i in tr_samples])
    cv_data = ([X[i] for i in cv_samples], [Y_encoded[i] for i in cv_samples])
    test_data = ([X[i] for i in test_samples], [Y_encoded[i] for i in test_samples])
    
    return (tr_data, cv_data, test_data)





    
    
    
    



                
        	



