import os
import sys
from pathlib import Path
import numpy as np
import cv2 as cv

# This script binarizes all images in the passed folder and stores them into a subfolder called "binarized"
# As threshold, the average pixel value is calculated.
# usage: py binarize_images.py ./your/folder/path

folder = sys.argv[1]

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def binarizeImage(filename):
    im = cv.imread(folder + '/' + filename)
    Path(folder + "/binarized").mkdir(parents=True, exist_ok=True)

    avg_color_per_row = np.average(im, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    th, im_binarized = cv.threshold(im, avg_color[0], np.iinfo(im.dtype).max, cv.THRESH_BINARY)
    cv.imwrite(folder + "/binarized/" + f, im_binarized)


for f in files(folder):
    binarizeImage(f)
