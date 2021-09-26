import glob
import sys
import numpy as np
import cv2 as cv
from numpy import asarray

# This script implements a fusion of multiple images.
# It is NOT used for any other part and also not used in the thesis. This is only experimental.
#
# usage: py msi_fusion.py <mode> ./your/folder/path/startname ./output/path
# mode: currently, only "overlay" exists

mode = sys.argv[1]
msiStartPath = sys.argv[2]
outputPath = sys.argv[3]

def overlayFusion(msi):
    print(msi)
    weight = 1.0 / len(msi)
    currentImage = cv.imread(msi.pop())

    overlayImage = np.zeros_like(currentImage)
    overlayImage = overlayImage + currentImage * weight

    for filename in msi:
        nextImage = cv.imread(filename)
        overlayImage = overlayImage + nextImage * weight

    overlayImage = overlayImage.astype(np.uint8)

    #while len(msi) != 0:
        #nextImageFilename = msi.pop()
        #nextImage = cv.imread(nextImageFilename)
        #print(nextImageFilename)

        #overlayImage = cv.addWeighted(overlayImage, weight, nextImage, weight, 0)
        #mask1 = (nextImage > 127.5)
        #mask2 = (nextImage <= 127.5)

        #np.putmask(overlayImage, mask1, 255 - overlayImage)
        #np.putmask(nextImage, mask1, 255 - nextImage)
        #overlayImage[mask1] = 255 - overlayImage[mask1]
        #nextImage[mask1] = 255 - nextImage[mask1]
        #overlayImage[mask1] = np.multiply(overlayImage[mask1], nextImage[mask1])
        #overlayImage[mask1] = overlayImage[mask1] / 127.5
        #overlayImage[mask1] = 255 - overlayImage[mask1]
        #np.putmask(overlayImage, mask1, 255 - overlayImage)

        #overlayImage[mask2] = np.multiply(overlayImage[mask2], nextImage[mask2])

        # np.putmask(overlayImage, mask, 255 - (((255 - overlayImage) * (255 - nextImage[mask])) / 127.5))

        #overlayImage[overlayImage > 127.5] = 255 - (((255 - np.exp(overlayImage[:, :])) * (255 - np.exp(nextImage[:, :]))) / 127.5)
        #overlayImage[overlayImage <= 127.5] = np.exp(overlayImage[:, :]) * np.exp(nextImage[:, :])

    cv.imwrite(outputPath + '/' + msiStartPath.split('/')[-1] + '_' + mode + '.png', overlayImage)


msiList = glob.glob(msiStartPath + '*')
msiList.sort(key=len)
print("Mode: " + mode)

if mode == 'overlay':
    overlayFusion(msiList)
else:
    print("Unknown mode!")
