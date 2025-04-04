import cv2 as cv
import numpy as np


def Sift(train, test, key_or_desc):
    '''
        Extract SIFT features from the input img.
        Variable:
            train(list): List of training data
            test(list): List of testing data
            key_or_desc(str): choosing to return keypoints or descriptors ('key', 'desc')
        Returns:
            train_sift_features(list): a list of all SIFT features from train data.
            test_sift_features(list): a list of all SIFT features from test data.
    '''
    ## Create SIFT Object and Compute Keypoints and Descriptors ##
    train_sift_features = []
    test_sift_features = []

    for img in train:
        sift = cv.SIFT_create()
        keypoints_t, descriptors_t = sift.detectAndCompute(img, None)
        if key_or_desc == 'key':
            train_sift_features.append(keypoints_t)
        if key_or_desc == 'desc':
            train_sift_features.append(descriptors_t)
    
    for img in test:
        sift = cv.SIFT_create()
        keypoints_v, descriptors_v = sift.detectAndCompute(img, None)
        if key_or_desc == 'key':
            test_sift_features.append(keypoints_v)
        if key_or_desc == 'desc':
            test_sift_features.append(descriptors_v)
    
    return train_sift_features, test_sift_features


def HoG(train, test):
    '''
        Extract HoG features from the input img. Using L2 Normalization.
        Variable:
            train(list): List of training data
            test(list): List of testing data
        Returns:
            Returns:
            train_sift_features(list): a list of all Hog features from train data.
            test_sift_features(list): a list of all HoG features from test data.
    '''
    img_size = train[0].shape # h x w in pixels <- take a random image
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9  # number of orientation bins

    # create HoG Object
    hog = cv.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1], 
                                     img_size[0] // cell_size[0] * cell_size[0]), 
                           _blockSize=(block_size[1] * cell_size[1], 
                                       block_size[0] * cell_size[0]),
                           _blockStride=(cell_size[1], cell_size[0]),
                           _cellSize=(cell_size[1], cell_size[0]),
                           _nbins=nbins,
                           _histogramNormType= cv.NORM_L2)
    
    train_hog_features = []
    test_hog_features = []
    ## Compute Features ##
    for img in train:
        # Hog Features
        temp = hog.compute(img.astype(np.uint8))
        train_hog_features.append(temp / np.linalg.norm(temp, ord=2)) # L2 Normalization
    
    for img in test:
        # Hog Features
        temp = hog.compute(img.astype(np.uint8))
        test_hog_features.append(temp / np.linalg.norm(temp, ord=2)) # L2 Normalization
    
    return train_hog_features, test_hog_features