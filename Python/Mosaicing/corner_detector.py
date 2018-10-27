'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from skimage.feature import corner_harris, corner_peaks
import numpy as np
import cv2
import matplotlib.pyplot as plt


def corner_detector(img):
    cimg = corner_harris(img)
    return cimg


if __name__ == "__main__":

    left = cv2.imread("../test_img/1L.jpg")

    # Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    # Get the corner metrics for each pixel
    corner_metric_matrix = corner_detector(gray_left)

    # Plotting corners on the gray scale image
    corners = corner_peaks(corner_metric_matrix)
    fig, ax = plt.subplots()
    ax.imshow(gray_left, origin='upper', cmap=plt.cm.gray)
    ax.plot(corners[:,1], corners[:, 0], '+r', markersize=15)
    h, w = gray_left.shape

    plt.show()

