import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib.path as mplpath
from PIL import Image, ImageDraw



def stitch(im1, im2, H, addXOff = 0):
    im1w, corners, xoff, yoff = stitch2(im1, im2, H)
    corners = np.array([corners[1], corners[0]])
    h2, w2, d = im2.shape
    im2Corners = np.array([[0, h2, h2, 0], [0, 0, w2, w2]])
    allCorns = np.hstack((corners, im2Corners)).T
    print(corners)
    print(im2Corners)
    print(allCorns)

    #I am ashamed of the next chunk of code
    bottomPts   = allCorns[[3, 2, 5, 6]]
    topPts      = allCorns[[0, 1, 4, 7]]    
    leftPts     = allCorns[[0, 3, 4, 5]]
    rightPts    = allCorns[[1, 2, 7, 6]]

    if xoff < 0:
        canvas = np.zeros((im2.shape[0], int(im2.shape[1] - xoff) , 3))    
        canvas[:, -1 * int(xoff) + addXOff :, :] = im2.astype(int)
        print(yoff)
        canvas[:, :int(im1w.shape[1]), :] += im1w[int(-1 * yoff) : int(-1 * yoff + canvas.shape[0]), :, :]
        canvas[:, -1 * int(xoff) + addXOff : im1w.shape[1], : ] *= 0.5 
        
    else:
        canvas = np.zeros((im2.shape[0], int(im2.shape[1] - xoff + im1w.shape[1]), 3))#todo: right-to-left

    plt.plot()
    plt.imshow(canvas[:, :, [2, 1, 0]].astype(int))
    plt.show()


    return canvas, min(-1 * xoff, 0)


def stitch2(im1, im2, H):
    """
    @return Image of the warped im1
    @return corner locations of im1 in the im2 space
    
    """

    from mymosaic import interpolateImg
    Hinv = np.linalg.inv(H)

    h1, w1, d = im1.shape
    h2, w2, d = im2.shape

    im1Corners = np.array([[0, w1, w1, 0], [0, 0, h1, h1], [1,1,1,1]])
    im1Corners = im1Corners.T.reshape(4,3,1)
   # print(im1Corners)
    im1CornersS = np.squeeze(np.matmul(H, im1Corners))
    im1CornersS = (im1CornersS[:, :] / im1CornersS[:, [-1]]).T[:-1, :]
   # print(im1CornersS)
    im2Corners = np.array([[0, w2, w2, 0], [0, 0, h2, h2], [1,1,1,1]])
    im2Corners = im2Corners.T.reshape(4,3,1)
   # print(im2Corners)
    im2CornersS = np.squeeze(np.matmul(np.linalg.inv(H), im2Corners))
    im2CornersS = (im2CornersS[:, :] / im2CornersS[:, [-1]]).T[:-1, :]
   # print(im2CornersS)


    posits = np.mgrid[0:w1, 0:h1].reshape(2, w1 * h1)#[x coords, y coords]
    posits = np.vstack((posits, np.ones(w1 * h1))).T.reshape(w1*h1, 3, 1)
    posits = np.mgrid[0:w1, 0:h1].reshape(2, w1 * h1)#[x coords, y coords]
    posits = np.vstack((posits, np.ones(w1 * h1))).T.reshape(w1*h1, 3, 1)

    newPosits = np.squeeze(np.matmul(H, posits))
    newPosits = newPosits[:, :] / newPosits[:, [-1]]
    newPosits = np.delete(newPosits, 2, axis=1)
    
    xmin = np.amin(newPosits[:, :-1])
    xmax = np.amax(newPosits[:, :-1])
    ymin = np.amin(newPosits[:,1:])
    ymax = np.amax(newPosits[:,1:])
    

    xRange = int(math.ceil(xmax - xmin))
    yRange = int(math.ceil(ymax - ymin))

    interpX, interpY = np.meshgrid(np.arange(xmin, xmax, dtype=int), np.arange(ymin, ymax, dtype = int), indexing='ij')
    interpX = interpX.flatten()
    interpY = interpY.flatten()
    interpPts = np.array([interpX, interpY, np.ones(interpX.size)]).T.reshape(interpX.size, 3, 1)
    Hmult_result = np.matmul(Hinv, interpPts)
    Hmult_result = Hmult_result[:, :] / Hmult_result[:, [-1]]
    Hmult_result.squeeze()
    Hmult_result = np.fliplr(Hmult_result[:, :-1]).squeeze()

    newInterp = interpolateImg(Hmult_result.squeeze(), im1)
    badwarp = newInterp.reshape((xRange, yRange, 3)).astype(int)
    badwarp = badwarp.swapaxes(0,1)
    
    plotX = im1CornersS[0] - xmin
    plotY = im1CornersS[1] - ymin
    
    plot2X = im1CornersS[0]
    plot2Y = im1CornersS[1]

    coords = np.vstack((plotX, plotY)).T.astype(int)

    mypath = mplpath.Path(im1CornersS.T.astype(int))

    """
    wholeCanvas = np.zeros((yRange, xRange , 3))
    wholeCanvas[int(ymin): int(ymin) + im2.shape[0], int(xmin):int(xmin) + im2.shape[1] , :] = im2[:, :, :]
    wholeCanvas = wholeCanvas.astype(int)
    """

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(badwarp[:, :, [2,1,0]], aspect="equal")
    ax[0].scatter(x=plotX, y=plotY, color='blue')
    ax[1].imshow(im2[:, :, [2,1,0]], aspect="equal")
    ax[1].scatter(x=plot2X, y=plot2Y, color='red')
    plt.show()

    return badwarp, im1CornersS, xmin, ymin




