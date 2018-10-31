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
RSAC_MIN_CONSENSUS  = 10


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
        while (len(indexex) < 4):
            x = int(np.random.random() * n)
            if x not in indexes:
                indexes.append(x)
        #at this point, indexes has 4 random unique numbers in it
        indexes = np.array(indexes)
        H = hFrom8Points(x1[indexes], y1[indexes], x2[indexes], y2[indexes])

        origIndexes = np.array([x1, y1, np.ones(n, dtype = float)]).T.reshape((n, 3, 1))
        origIndexes = np.delete(origIndexes, indexes, axis = 0)

        multByH = np.matmul(H, origIndexes)

        multByH = np.delete(multByH, 2, axis = 1)
        multByH = multByH.reshape((n, 2)).T

        newx1 = multByH[0]
        newy1 = multByH[1]

        dist = np.sqrt(np.sum( ((x2 - newx1) * (x2 - newx1)), ((y2 - newy1) * (y2 - newy1))))

        isUnderDist = np.less_equal(dist, thresh)
        goodEnoughCount = np.count_nonzero(isUnderDist)

        if goodEnoughCount > RSAC_MIN_CONSENSUS:
            t += 1
            if goodEnoughCount > bestConsensus:
                bestH = H
                bestN = isUnderDict.astype(int)
                bestConsensus = goodEnoughCount

    
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








