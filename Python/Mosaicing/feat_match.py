'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc


def feat_match(descs1, descs2):
    rows, num_points_in_1 = descs1.shape

    match = -1*np.ones((num_points_in_1, 1))

    # Initialize the KDTree
    # Have to transpose because the KDTree expects the points ot be rows, not columns
    tree = KDTree(descs2.T)
    temp_descs1 = descs1.T

    # Setting k=2 asks the tree for the two closest neighbors
    distances, indexes = tree.query(temp_descs1, k=2)

    # Vectorized
    ratios = distances[:,0] / distances[:,1]
    mask = ratios < 0.7
    num_matches = len(indexes[:,0][mask])
    match[mask] = indexes[:,0][mask].reshape(num_matches, 1)





    return match

if __name__ == "__main__":

    left = cv2.imread("../test_img/small_1L.jpg")
    middle = cv2.imread("../test_img/small_1M.jpg")

    # Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_middle = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)

    left_cmm = corner_detector(gray_left)
    middle_cmm = corner_detector(gray_middle)

    features = 1000
    left_c, left_r, left_rmax = anms(left_cmm, features)
    middle_c, middle_r, middle_rmax = anms(middle_cmm, features)

    # Show the points for the left
    # fig, ax = plt.subplots()
    # ax.imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    # ax.plot(left_c, left_r, 'r.', markersize=5)

    # plt.show()

    # Show the points for the middle
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    ax1.plot(left_c, left_r, 'r.', markersize=5)
    ax2.imshow(gray_middle, origin='upper', cmap=plt.cm.gray)
    ax2.plot(middle_c, middle_r, 'r.', markersize=5)

    plt.show()


    left_descs = feat_desc(gray_left, left_c, left_r)
    middle_descs = feat_desc(gray_middle, middle_c, middle_r)

    matches = feat_match(left_descs, middle_descs)
    print(matches)
