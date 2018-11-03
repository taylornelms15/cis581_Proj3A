'''
  File name: anms.py
  Author: Taylor Nelms
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
from skimage.feature import corner_harris, corner_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt
from corner_detector import corner_detector
from scipy.spatial import distance
from scipy import stats
import time

def anms(cimg, max_pts):
    # Keep track of the minimum distance to larger magnitude feature point
    points = list()
    """
    cmax = np.amax(cimg)
    cmin = np.amin(cimg)
    vrange = cmax - cmin
    dimg = cimg - cmin
    dimg = dimg * 1.0/ vrange
    dimg *= 255
    dimg = dimg.astype(int)
    plt.figure()
    plt.imshow(dimg)
    plt.show()
    """

    '''
    For each feature point:
        Find all the feature points with a magnitude more than .9 of the pixel under consideration
        Of the remaining pixels, find the minimum distance 
    '''

    h, w = cimg.shape
    cimg[cimg <= 0] = 0
    r, c = np.where(cimg > 0)
    good_points = cimg[cimg > 0]
    mag_and_index = [(good_points[i], (r[i], c[i])) for i in range(len(good_points))]
    mag_and_index.sort(key=lambda x: x[0], reverse=True)
    
    if len(mag_and_index) >= 30000:
        # Take the most cornery points
        mag_and_index = mag_and_index[0:30000]


    mag = np.array([x[0] for x in mag_and_index])
    x = np.array([x[1][0] for x in mag_and_index])
    x = x.reshape(len(x),1)
    y = np.array([x[1][1] for x in mag_and_index])
    y = y.reshape(len(y),1)

    index = np.hstack((x,y))


    best_distances = list()
    num_candidates = len(mag_and_index)
    for i in range(num_candidates):
        cur_point = index[i]
        cur_mag = cimg[index[i][0], index[i][1]]

        mag_candidates = mag[cur_mag < 0.9*mag]
        index_candidates = index[0:mag_candidates.shape[0]]

        if len(mag_candidates) != 0 and len(index_candidates) != 0:

            distances = np.sqrt(np.sum(np.power(index_candidates - cur_point, 2), axis=1))

            best_dist = np.min(distances)
            best_point = index_candidates[np.argmin(distances)]

            best_distances.append((best_dist, (best_point[0], best_point[1])))

        

    # Sort the list in descending order of distance
    best_distances.sort(key=lambda x: x[0], reverse=True)

    top_points = np.array([x[1] for x in best_distances[0:max_pts]])
    x = top_points[:,1] # Represents column coordinates
    y = top_points[:,0] # Represents row coordinates

    # rmax is the minimum distance away the point must be to make it into the corner set
    # Not sure about this.
    rmax = best_distances[max_pts-1][0]

    return x, y, rmax

if __name__ == "__main__":

    # left = cv2.imread("../test_img/small_1L.jpg")
    # midd = cv2.imread("../test_img/small_1M.jpg")

    left = cv2.imread("../test_img/pano1.jpg")
    midd = cv2.imread("../test_img/pano2.jpg")
    right = cv2.imread("../test_img/pano3.jpg")

    # Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_midd = cv2.cvtColor(midd, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Get the corner metrics for each pixel
    cimgL = corner_detector(gray_left)
    cimgM = corner_detector(gray_midd)
    cimgR = corner_detector(gray_right)

    xL, yL, rmaxL = anms(cimgL, 1000)
    xM, yM, rmaxM = anms(cimgM, 1000)
    xR, yR, rmaxR = anms(cimgR, 1000)

    # Plotting corners on the gray scale image

    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    # ax[0].plot(xL, yL, '.r', markersize=5)
    # ax[1].imshow(gray_midd, origin='upper', cmap=plt.cm.gray)
    # ax[1].plot(xM, yM, '.r', markersize=5)
    # h, w = gray_left.shape

    # plt.show()

    fig, ax = plt.subplots(ncols=1)
    ax.imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    ax.plot(xL, yL, '.r', markersize=5)
    plt.show()

    fig, ax = plt.subplots(ncols=1)
    ax.imshow(gray_midd, origin='upper', cmap=plt.cm.gray)
    ax.plot(xM, yM, '.r', markersize=5)
    plt.show()

    fig, ax = plt.subplots(ncols=1)
    ax.imshow(gray_right, origin='upper', cmap=plt.cm.gray)
    ax.plot(xR, yR, '.r', markersize=5)
    plt.show()








