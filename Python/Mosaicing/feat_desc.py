'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt


def feat_desc(img, x, y):
    
    # x is the column coordinate
    # y is the row coordinate

    # Mirroring edge pixels to get boundary
    padded_img = img
    r, c = img.shape

    # pad the top
    temp = np.ones((1,c))
    for i in range(20):
        temp = np.vstack((img[i,:], temp))
    
    temp = temp[0:20,:]
    padded_img = np.vstack((temp, padded_img))


    # Pad the bottom
    temp = np.ones((1,c))
    for i in range(r-20,r):
        temp = np.vstack((img[i,:], temp))

    temp = temp[0:20,:]
    padded_img = np.vstack((padded_img, temp))

    r, c = padded_img.shape
    # Pad the left side
    temp = np.ones((r, 1))
    for i in range(20):
        temp = np.hstack((padded_img[:,i].reshape(r,1), temp))

    temp = temp[:,0:20]
    padded_img = np.hstack((temp, padded_img))

    # Pad the right side
    temp = np.ones((r, 1))
    for i in range(c-20, c):
        temp = np.hstack((padded_img[:,i].reshape(r,1), temp))

    temp = temp[:,0:20]
    padded_img = np.hstack((padded_img, temp))


    x = x + 20
    y = y + 20
    descs = np.ones((64,1))
    for i in range(len(x)):
        big_patch = padded_img[y[i]-20:y[i]+20, x[i]-20:x[i]+20]
        small_patch = big_patch[0:40:5,0:40:5]
        normalized_small_patch = (small_patch - np.mean(small_patch)) / np.std(small_patch)
        normalized_small_patch = normalized_small_patch.reshape(64,1)
        descs = np.hstack((descs, normalized_small_patch))


    descs = descs[:, 1:]

    return descs


if __name__ == "__main__":

    img = cv2.imread("../test_img/small_1L.jpg")
    # Convert to grayscale
    gray_left = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the corner metrics for each pixel
    cimg = corner_detector(gray_left)

    x, y, rmax = anms(cimg, 500)

    descs = feat_desc(gray_left, x, y)

