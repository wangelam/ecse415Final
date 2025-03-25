
import os
import cv2 as cv
import numpy as np

def mask_preprocess(image_name, mask_path):
    '''
        For mask preprocessing.
        Variables:
            image_name(str): name of the image including the file name extension
            mask_path(str): full path to the folder containing the mask
        Return:
            mask(numpy.ndarray): the cleaned labeled mask
            binary(numpy.ndarray): the cleaned binary mask

    '''
    mask = cv.imread(os.path.join(mask_path, image_name), cv.IMREAD_GRAYSCALE)
    mask = np.where(mask == np.unique(mask)[-1], 0, mask) # remove red bounding box
    _, binary = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    return mask, binary

def dice_coefficient(img1, img2):
    intersection = cv.bitwise_and(img1, img2)
    union = cv.bitwise_or(img1, img2)
    dice = (2*np.sum(intersection)) / (np.sum(union)+np.sum(intersection))
    return dice


def mask_binarize(image):
    if(len(image.shape)>2):
        img_b = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_b = image

    # Finding the unique values in the grayscale img
    thresh = np.unique(img_b)[0]
    _, img_b = cv.threshold(img_b, thresh, 255, cv.THRESH_BINARY_INV)
    return img_b

