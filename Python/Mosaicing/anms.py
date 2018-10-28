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

def anms(cimg, max_pts):
    corners = corner_peaks(cimg)

    if max_pts > corners.shape[0]:
        # More points than are returned
        print("Error: Number of corners less than the number of max points")
        exit(1)

    # Keep track of the minimum distance to larger magnitude feature point
    points = list()

    '''
    For each feature point:
        Find all the feature points with a magnitude more than .9 of the pixel under consideration
        Of the remaining pixels, find the minimum distance 
    '''
    h, w = cimg.shape

    for i in range(h):
        print("Working on row {0}".format(i))
        for j in range(w):
            mask = (cimg > .9*cimg[i,j]).astype(int)
            points2comp = cimg*mask

            # Get the others
            x, y = np.where(points2comp > 0)

            # Get the distance
            # pairwise = distance.cdist(source, source, 'euclidean')
            distances = np.sqrt(np.power(x-i,2) + np.power(y-j,2))

            # Get the minimum
            # The only zero would be when comparing a pixel to itself, so we consider everything else
            # print("\tlen(distances) = {0}".format(len(distances)))
            if len(distances) == 1:
                min_dist = 0
            else:
                min_dist = np.min(distances[distances > 0])

            # Append a tuple of the pixel coordinates and the min distance to the nearest good feature point
            points.append(((i,j), min_dist))


    # Sort the list in descending order of distance
    points.sort(key=lambda x: x[1], reverse=True)

    top_points = np.array([x[0] for x in points[0:max_pts]])
    x = top_points[:,1] # Represents column coordinates
    y = top_points[:,0] # Represents row coordinates

    # rmax is the minimum distance away the point must be to make it into the corner set
    # Not sure about this.
    rmax = points[0:max_pts][-1][1]

    return x, y, rmax

if __name__ == "__main__":

    left = cv2.imread("../test_img/small_1L.jpg")

    # Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    # Get the corner metrics for each pixel
    corner_metric_matrix = corner_detector(gray_left)

    x, y, rmax = anms(corner_metric_matrix, 10)
    # Plotting corners on the gray scale image
    # corners = corner_peaks(corner_metric_matrix)
    # fig, ax = plt.subplots()
    # ax.imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    # ax.plot(corners[:,1], corners[:, 0], '+r', markersize=15)
    # h, w = gray_left.shape

    # plt.show()

