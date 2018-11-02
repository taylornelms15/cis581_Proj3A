import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib.path as mplpath
import sys
import traceback



def stitch(im1, im2, H, addXOff = 0):
    im1w, overlap, xOffset = stitchL(im1, im2, H)

    try:
        canvas = np.zeros((im2.shape[0], int(im2.shape[1] - xOffset), 3))
        canvas[:, : -xOffset, :] = im1w[:, : -overlap, :]
        canvas[:, -xOffset : -xOffset + overlap, :] = 0.5 * im1w[:, -xOffset : , :]
        canvas[:, -xOffset : -xOffset + overlap, :] += 0.5 * im2[:, :overlap, :]
        canvas[:, -xOffset + overlap : , :] = im2[:, overlap: , :]
    except Exception as e:
        summary = traceback.StackSummary.extract(
            traceback.walk_stack(None)
        )
        print(''.join(summary.format()))

    """
    plt.plot()
    plt.imshow(canvas[:, :, [2, 1, 0]].astype(int))
    plt.show()
    """
    return canvas, xOffset



def stitchRR(im1, im2, H, addXOff = 0, leftSide = None):
    im1w, overlap, im1Width = stitchR(im1, im2, H,)

    canvas = np.zeros((im2.shape[0], leftSide.shape[1] + im1w.shape[1] - overlap, 3))
    canvas[:, : leftSide.shape[1] - overlap, :] = leftSide[:, : -overlap, :]
    canvas[:, leftSide.shape[1] - overlap : leftSide.shape[1], :] = 0.5 * leftSide[:, -overlap : , :]
    canvas[:, leftSide.shape[1] - overlap : leftSide.shape[1], :] += 0.5 * im1w[:, :overlap, :]
    canvas[:, leftSide.shape[1] : , :] = im1w[:, overlap: , :]

    """
    plt.plot()
    plt.imshow(canvas[:, :, [2, 1, 0]].astype(int))
    plt.show()
    """

    return canvas
    """
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
    """


def stitchL(im1, im2, H):
    """
    @return Image of the warped im1
    @return amount of overlap
    @return offset for our im1 (negative)
    """

    from mymosaic import interpolateImg
    Hinv = np.linalg.inv(H)

    h1, w1, d = im1.shape

    im1Corners = np.array([[0, w1, w1, 0], [0, 0, h1, h1], [1,1,1,1]])
    im1Corners = im1Corners.T.reshape(4,3,1)
    #print(im1Corners)
    im1CornersS = np.squeeze(np.matmul(H, im1Corners))
    im1CornersS = (im1CornersS[:, :] / im1CornersS[:, [-1]]).T[:-1, :]
    #print(im1CornersS)
    xmin = np.amin(im1CornersS[0])
    xmax = np.amax(im1CornersS[0])
    ymin = np.amin(im1CornersS[1])
    ymax = np.amax(im1CornersS[1])
    
    xRange = int(math.ceil(xmax - xmin))
    yRange = int(math.ceil(ymax - ymin))

    xminR = int(math.floor(xmin))
    xmaxR = int(math.ceil(xmax))
    yminR = int(math.floor(ymin))
    ymaxR = int(math.ceil(ymax))

    interpX, interpY = np.meshgrid(np.arange(xminR, xmaxR, dtype=int), np.arange(0, im2.shape[0], dtype = int), indexing='ij')
    interpX = interpX.flatten()
    interpY = interpY.flatten()
    interpPts = np.array([interpX, interpY, np.ones(interpX.size)]).T.reshape(interpX.size, 3, 1)
    Hmult_result = np.matmul(Hinv, interpPts)
    Hmult_result = Hmult_result[:, :] / Hmult_result[:, [-1]]
    Hmult_result.squeeze()
    Hmult_result = np.fliplr(Hmult_result[:, :-1]).squeeze()

    newInterp = interpolateImg(Hmult_result.squeeze(), im1)
    badwarp = newInterp.reshape((xmaxR - xminR, im2.shape[0], 3)).astype(int)
    badwarp = badwarp.swapaxes(0,1)
    return badwarp, xmaxR, xminR

    """
    plt.plot()
    plt.imshow(badwarp[:, :, [2, 1, 0]])
    plt.show()
    
    plotX = im1CornersS[0] - xmin
    plotY = im1CornersS[1]
    
    plot2X = im1CornersS[0]
    plot2Y = im1CornersS[1]

    coords = np.vstack((plotX, plotY)).T.astype(int)


    wholeCanvas = np.zeros((yRange, xRange , 3))
    wholeCanvas[int(ymin): int(ymin) + im2.shape[0], int(xmin):int(xmin) + im2.shape[1] , :] = im2[:, :, :]
    wholeCanvas = wholeCanvas.astype(int)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(badwarp[:, :, [2,1,0]], aspect="equal")
    ax[0].scatter(x=plotX, y=plotY, color='blue')
    ax[1].imshow(im2[:, :, [2,1,0]], aspect="equal")
    ax[1].scatter(x=plot2X, y=plot2Y, color='red')
    plt.show()

    return badwarp, xmaxR, xminR
    """

def stitchR(im1, im2, H):
    """
    @return Image of the warped im1
    @return amount of overlap
    @return width of our im1 at the end
    """

    from mymosaic import interpolateImg
    Hinv = np.linalg.inv(H)

    h1, w1, d = im1.shape

    im1Corners = np.array([[0, w1, w1, 0], [0, 0, h1, h1], [1,1,1,1]])
    im1Corners = im1Corners.T.reshape(4,3,1)
    #print(im1Corners)
    im1CornersS = np.squeeze(np.matmul(H, im1Corners))
    im1CornersS = (im1CornersS[:, :] / im1CornersS[:, [-1]]).T[:-1, :]
    #print(im1CornersS)
    xmin = np.amin(im1CornersS[0])
    xmax = np.amax(im1CornersS[0])
    ymin = np.amin(im1CornersS[1])
    ymax = np.amax(im1CornersS[1])
    
    xRange = int(math.ceil(xmax - xmin))
    yRange = int(math.ceil(ymax - ymin))

    xminR = int(math.floor(xmin))
    xmaxR = int(math.ceil(xmax))
    yminR = int(math.floor(ymin))
    ymaxR = int(math.ceil(ymax))

    interpX, interpY = np.meshgrid(np.arange(xminR, xmaxR, dtype=int), np.arange(0, im2.shape[0], dtype = int), indexing='ij')
    interpX = interpX.flatten()
    interpY = interpY.flatten()
    interpPts = np.array([interpX, interpY, np.ones(interpX.size)]).T.reshape(interpX.size, 3, 1)
    Hmult_result = np.matmul(Hinv, interpPts)
    Hmult_result = Hmult_result[:, :] / Hmult_result[:, [-1]]
    Hmult_result.squeeze()
    Hmult_result = np.fliplr(Hmult_result[:, :-1]).squeeze()

    newInterp = interpolateImg(Hmult_result.squeeze(), im1)
    badwarp = newInterp.reshape((xmaxR - xminR, im2.shape[0], 3)).astype(int)
    badwarp = badwarp.swapaxes(0,1)
    return badwarp, im2.shape[1] - xminR, xmaxR - xminR



