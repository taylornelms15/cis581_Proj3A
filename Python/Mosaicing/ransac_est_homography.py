'''
  File name: ransac_est_homography.py
  Author: Taylor Nelms
  Date created: 10/30/2018
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import matplotlib.pyplot as plt
from est_homography import est_homography
from scipy.spatial.distance import euclidean



RSAC_NUM_TRIALS     = 1000
RSAC_MIN_CONSENSUS  = 3


def ransac_est_homography(x1, y1, x2, y2, thresh):
    """
    @return: Indexes of our best four points
    """
    n = len(x1)

    bestH = np.zeros((3,3,3))
    bestN = np.zeros(n)
    bestConsensus = 0
    
    t = 0
    while t < RSAC_NUM_TRIALS:
        indexes = []
        while (len(indexes) < 4):
            x = int(np.random.random() * n)
            if x not in indexes:
                indexes.append(x)
        #at this point, indexes has 4 random unique numbers in it
        indexes = np.array(indexes)
        H = hFrom8Points(x1[indexes], y1[indexes], x2[indexes], y2[indexes])
        """
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(im1, origin='upper')
        ax[1].imshow(im2, origin='upper')
        ax[0].plot(x1[indexes], y1[indexes],  markersize=5, color='blue')
        ax[1].plot(x2[indexes], y2[indexes],  markersize=5, color='blue')
        """


        origIndexes = np.array([x1, y1, np.ones(n, dtype = float)]).T.reshape((n, 3, 1))
        origIndexesMod = np.delete(origIndexes, indexes, axis = 0)

        multByH = np.matmul(H, origIndexesMod)
        multByH = np.squeeze(multByH)
        multByH = multByH[:,:] / multByH[:, [-1]]

        multByH = np.delete(multByH, 2, axis = 1)
        multByH = multByH.reshape((n-4, 2)).T

        newx1 = multByH[0]
        newy1 = multByH[1]
        newx2 = np.delete(x2, indexes, axis=0)
        newy2 = np.delete(y2, indexes, axis=0)
        """
        ax[1].plot(np.delete(x2, indexes), np.delete(y2, indexes),'.r', markersize=3, color='red')
        ax[1].plot(newx2, newy2,'.r', markersize=3, color='yellow')

        plt.show()
        """
        distx = newx2 - newx1
        disty = newy2 - newy1

        dist = np.sqrt(distx * distx + disty * disty)

        isUnderDist = np.less_equal(dist, thresh)
        indexes = np.sort(indexes)
        for i in range(len(indexes)):
            isUnderDist = np.insert(isUnderDist, indexes[i], False)
        goodEnoughCount = np.count_nonzero(isUnderDist)


        if goodEnoughCount >= bestConsensus:
            bestH = H
            bestN = isUnderDist.astype(int)
            bestConsensus = goodEnoughCount
        t+=1

    
    return bestH, bestN



def hFrom8Points(x1, y1, x2, y2):
    """
    @param x1: 4 x coordinates for our source points 
    @param y1: 4 y coordinates for our source points 
    @param x2: 4 x coordinates for our dest points 
    @param y2: 4 y coordinates for our dest points 
    @return: Our 3x3 H matrix
    """

    return est_homography(x1, y1, x2, y2)


def main():
    pass



if __name__ == "__main__":
    main()








