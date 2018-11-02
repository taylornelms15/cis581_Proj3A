'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input imgs: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cv2
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from scipy import interpolate
import math
import stitcher
import sys
import traceback
from importlib import reload
from matplotlib.patches import ConnectionPatch
#import pdb

RSAC_THRESH_VAL = 3.0

def mymosaic(img_input):
    imgs = []
    for img in img_input:
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    cimg = []
    aNMS = []
    descs = []
    fmatchsing = []
    featMatchesDouble = []
    hMat = []

    for img in imgs:
        cimg.append(corner_detector(img))

    for i, c in enumerate(cimg):
        ares = anms(c, 1000)
        fig, ax = plt.subplots()
        ax.imshow(imgs[i], origin="upper", cmap=plt.cm.gray)
        ax.plot(ares[0], ares[1], '.r', markersize=2, color="red")
        plt.show()
        aNMS.append(ares)

    for i, aN in enumerate(aNMS):
        descs.append(feat_desc(imgs[i], aN[0], aN[1]))

    for i in range(len(descs) - 1):
        fDirect = feat_match(descs[i], descs[i + 1])
        bDirect = feat_match(descs[i + 1], descs[i])
        m1m = fDirect.T[0][(fDirect.T[0] != -1)].astype(int)
        m2m = bDirect.T[0][(bDirect.T[0] != -1)].astype(int)
        """
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(imgs[i], origin="upper", cmap=plt.cm.gray)
        ax[0].plot(aNMS[i][0], aNMS[i][1], '.r',  markersize=5, color='blue')
        ax[0].plot(aNMS[i][0][m2m], aNMS[i][1][m2m], '.r',  markersize=5, color='red')
        ax[1].imshow(imgs[i+1], origin="upper", cmap=plt.cm.gray)
        ax[1].plot(aNMS[i+1][0], aNMS[i+1][1], '.r', markersize=5, color='blue')
        ax[1].plot(aNMS[i+1][0][m1m], aNMS[i+1][1][m1m], '.r',  markersize=5, color='red')
        plt.show()
        """

        fmatchsing.append(fDirect)
        fmatchsing.append(bDirect)

    for i in range(int(len(fmatchsing) / 2)):


        srcindexes, dstindexes = unityOfMatch(fmatchsing[2 * i], fmatchsing[2 * i + 1])
        mX1 = aNMS[i][0][srcindexes]
        mY1 = aNMS[i][1][srcindexes]
        mX2 = aNMS[i+1][0][dstindexes]
        mY2 = aNMS[i+1][1][dstindexes]

        """
        if i < 2:
            m = np.zeros((imgs[i].shape[0], 2*imgs[i].shape[1], 3))
            m[0:imgs[i].shape[0], 0: imgs[i].shape[1],:] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)
            m[0:imgs[i].shape[0], imgs[i].shape[1]:, :] = cv2.cvtColor(imgs[i+1], cv2.COLOR_GRAY2RGB)

            # color_m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

            offset = imgs[i].shape[1]
            for j in range(len(mX1)):
                cv2.line(m, (mY1[j], mX1[j]), (mY2[j], offset + mX2[j]), (255,0,0), 2)

            cv2.imwrite("matches_{0}.png".format(i), m)
        """
        pts1 = np.vstack((mX1, mY1)).T
        pts2 = np.vstack((mX2, mY2)).T


        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(imgs[i], origin="upper", cmap=plt.cm.gray)
        ax[0].plot(aNMS[i][0], aNMS[i][1], '.r',  markersize=2, color='blue')
        ax[0].plot(mX1, mY1, '.r',  markersize=2, color='red')
        ax[1].imshow(imgs[i+1], origin="upper", cmap=plt.cm.gray)
        ax[1].plot(aNMS[i+1][0], aNMS[i+1][1], '.r', markersize=2, color='blue')
        ax[1].plot(mX2, mY2, '.r', markersize=2, color='red')
        ax[1].set_zorder(-1)

        for j in range(len(mX1)):
            con = ConnectionPatch(pts1[j], pts2[j], "data", "data", axesA=ax[0], axesB=ax[1],zorder = 0.5)
            ax[0].add_patch(con)

        plt.show()
        
        rsac_results = ransac_est_homography(mX1, mY1, mX2, mY2, RSAC_THRESH_VAL)

        filtSrc = srcindexes[np.where(rsac_results[1])]
        filtDst = dstindexes[np.where(rsac_results[1])]


        hMat.append(rsac_results)
        H = rsac_results[0]

        mX1 = aNMS[i][0][filtSrc]
        mY1 = aNMS[i][1][filtSrc]
        mX2 = aNMS[i+1][0][filtDst]
        mY2 = aNMS[i+1][1][filtDst]
        pts1 = np.vstack((mX1, mY1)).T
        pts2 = np.vstack((mX2, mY2)).T

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(imgs[i], origin="upper", cmap=plt.cm.gray)
        ax[0].plot(aNMS[i][0], aNMS[i][1], '.r',  markersize=2, color='blue')
        ax[0].plot(mX1, mY1, '.r',  markersize=2, color='red')
        ax[1].imshow(imgs[i+1], origin="upper", cmap=plt.cm.gray)
        ax[1].plot(aNMS[i+1][0], aNMS[i+1][1], '.r', markersize=2, color='blue')
        ax[1].plot(mX2, mY2, '.r', markersize=2, color='red')
        ax[1].set_zorder(-1)
        for j in range(len(mX1)):
            con = ConnectionPatch(pts1[j], pts2[j], "data", "data", axesA=ax[0], axesB=ax[1], zorder=0.5)
            ax[0].add_patch(con)
        plt.show()

        """ 
        im2 = cv2.warpPerspective(imgs[1], H, dsize=(200, 150))
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(imgs[i], origin="upper", cmap=plt.cm.gray)
        ax[1].imshow(im2, origin="upper", cmap=plt.cm.gray)
        plt.show()
        """
        
    #at this point hMat[i] is the left-to-right H-result for that pair

    imIndexes = len(img_input) - 1
    m = (int)(imIndexes / 2)#floor; 3 images->index 1, 2 images->index 0

    partialImages = []

    hXform = []
    for result in hMat:
        hXform.append(result[0])#all the H matrixes

    numAbove = imIndexes - m
    numBelow = m

    #while(True):
    #pdb.set_trace()
    #reload(stitcher)
    intermed1, offset = stitcher.stitch(img_input[0], img_input[1], hXform[0])
    img_mosaic = intermed1
    intermed2 = None
    if len(hXform) > 1:
        intermed2 = stitcher.stitchRR(img_input[2], img_input[1], np.linalg.inv(hXform[1]), offset, intermed1)
        img_mosaic = intermed2


    if(np.amax(img_mosaic) < 2):
        return img_mosaic
    else:
        return img_mosaic.astype(int)


def unityOfMatch(m1, m2):
    """
    @param m1 Nx1 list of index locations (or -1) for second image that match to first
    @param m2 Mx1 list of index locations (or -1) for first image that match to second:w
    @return (indexes of rel feats for im1, indexes of rel feats for im2)
    """

    m1m = ma.masked_values(m1.T[0], -1).astype(int)
    m2m = ma.masked_values(m2.T[0], -1).astype(int)

    mid = m1m[m2m]
    whereMatch = np.logical_and(np.equal(m1m[m2m], np.arange(m2m.size)), m2m)
    locOfGoods = np.where(np.logical_and(whereMatch, np.not_equal(m2m, -1)))[0].astype(int)

    return m2.T[0].astype(int)[locOfGoods], locOfGoods

def interpolateImg(pts, img):
    """
    @param pts Nx2 matrix of points (mgrid?)
    @param img JxIx3 array that is the image
    @return Nx3, matrix of values at that point (rgb)
    """
    chan1 = img[:, :, 0]
    chan2 = img[:, :, 1]
    chan3 = img[:, :, 2]

    interp1 = interpolate.RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), chan1, method="linear", bounds_error=False, fill_value=0.0)
    interp2 = interpolate.RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), chan2, method="linear", bounds_error=False, fill_value=0.0)
    interp3 = interpolate.RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), chan3, method="linear", bounds_error=False, fill_value=0.0)

    newPts = np.clip(pts, [0,0], [img.shape[0] - 1, img.shape[1] - 1])
    val1 = interp1(newPts)
    val2 = interp2(newPts)
    val3 = interp3(newPts)
    return np.dstack((val1, val2, val3))



def main():
    """
    x1 = np.array([-1, 4, 5, -1, 0, 1])
    x2 = np.array([4, -1, -1, 3, -1, 2, 3])
    a1 = np.array([1, 0, 0, 0, 0, 2, 0, 0])
    a2 = np.array([0, 0, 2, 0, 1, 0])
    i1, i2 = unityOfMatch(x1, x2)
    print(unityOfMatch(x1, x2))
    print(a1[i1])
    print(a2[i2])
    """

    #left = cv2.imread("../test_img/small_1L.jpg")
    #middle = cv2.imread("../test_img/small_1M.jpg")
    #right = cv2.imread("../test_img/small_1R.jpg")

    left = cv2.imread("../test_img/1L.jpg")
    middle = cv2.imread("../test_img/1M.jpg")
    right = cv2.imread("../test_img/1R.jpg")
    # Convert to grayscale
    """
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_middle = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    """

    imgMatrix = [left, middle, right]
#    imgMatrix = [gray_left, gray_middle]
#    imgMatrix = [gray_left, gray_middle, gray_right]
    results = mymosaic(imgMatrix)
    plt.figure()
    plt.imshow(results[:, :, [2, 1, 0]])
    plt.show()
#    print(mymosaic(imgMatrix))


if __name__ == "__main__":
    main()





