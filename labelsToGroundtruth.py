import glob
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

# This script converts all images in the passed folder from labels (background black)
# into groundtruth (background white) for the DE-GAN training.
# The converted images are saved in a subfolder called "groundtruth".
#
# usage: py labelsToGroundtruth.py ./your/folder/path

folder = sys.argv[1]

def convertImage(file):
    g = cv.imread(file)

    notUsedPixelMask = cv.inRange(g, np.array([254, 0, 0]), np.array([255, 0, 0]))
    notUsedPixelMask = cv.bitwise_not(notUsedPixelMask)

    fg2 = cv.inRange(g, np.array([121, 121, 121]), np.array([123, 123, 123]))
    g[fg2==255] = [255, 255, 255]

    g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)

    g = cv.bitwise_and(g, notUsedPixelMask)
    g = cv.bitwise_not(g)
    cv.imwrite(folder + "/groundtruth/" + os.path.basename(file), g)


def convert():
    files = glob.glob(folder + '/*.png')
    print("Found labels: " + str(files))

    Path(folder + "/groundtruth").mkdir(parents=True, exist_ok=True)

    for f in files:
        print("Label: " + os.path.basename(f))
        convertImage(f)

if __name__ == '__main__':
    convert()
