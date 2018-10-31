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
    - Input img_input: M elements numpy array or list, each element is a input image.
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

RSAC_THRESH_VAL = 3.0

def mymosaic(img_input):

    cimg = []
    aNMS = []
    descs = []
    fmatchsing = []
    featMatchesDouble = []
    hMat = []

    for img in img_input:
        cimg.append(corner_detector(img))

    for c in cimg:
        aNMS.append(anms(c, 800))

    for i, aN in enumerate(aNMS):
        descs.append(feat_desc(img_input[i], aN[0], aN[1]))

    for i in range(len(descs) - 1):
        fDirect = feat_match(descs[i], descs[i + 1])
        bDirect = feat_match(descs[i + 1], descs[i])
        m1m = fDirect.T[0][(fDirect.T[0] != -1)].astype(int)
        m2m = bDirect.T[0][(bDirect.T[0] != -1)].astype(int)
        """
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_input[i], origin="upper", cmap=plt.cm.gray)
        ax[0].plot(aNMS[i][0], aNMS[i][1], '.r',  markersize=5, color='blue')
        ax[0].plot(aNMS[i][0][m2m], aNMS[i][1][m2m], '.r',  markersize=5, color='red')
        ax[1].imshow(img_input[i+1], origin="upper", cmap=plt.cm.gray)
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
        
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_input[i], origin="upper", cmap=plt.cm.gray)
        ax[0].plot(aNMS[i][0], aNMS[i][1], '.r',  markersize=5, color='blue')
        ax[0].plot(mX1, mY1, '.r',  markersize=5, color='red')
        ax[1].imshow(img_input[i+1], origin="upper", cmap=plt.cm.gray)
        ax[1].plot(aNMS[i+1][0], aNMS[i+1][1], '.r', markersize=5, color='blue')
        ax[1].plot(mX2, mY2, '.r', markersize=5, color='red')
        plt.show()

        rsac_results = ransac_est_homography(mX1, mY1, mX2, mY2, RSAC_THRESH_VAL)
        print(rsac_results)
        hMat.append(rsac_results)
        H = rsac_results[0]

        print(H)
        im2 = cv2.warpPerspective(img_input[1], H, dsize=(800, 800))

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img_input[i], origin="upper", cmap=plt.cm.gray)
        ax[1].imshow(im2, origin="upper", cmap=plt.cm.gray)
        plt.show()



    print(hMat)

    return img_mosaic


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

    left = cv2.imread("../test_img/small_1L.jpg")
    middle = cv2.imread("../test_img/small_1M.jpg")

    # Convert to grayscale
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_middle = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)

    imgMatrix = [gray_left, gray_middle]
    print(mymosaic(imgMatrix))


if __name__ == "__main__":
    main()





