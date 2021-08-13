import os
import sys
from pathlib import Path
import numpy as np
import cv2 as cv

folder = sys.argv[1]

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def binarizeImage(filename):
    im = cv.imread(folder + '/' + filename)
    #im = im.convert('L')
    #im = im.point(lambda x: 0 if x < 128 else 255, '1')
    Path(folder + "/binarized").mkdir(parents=True, exist_ok=True)

    avg_color_per_row = np.average(im, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    th, im_binarized = cv.threshold(im, avg_color[0], np.iinfo(im.dtype).max, cv.THRESH_BINARY)
    cv.imwrite(folder + "/binarized/" + f, im_binarized)
    #im.save(folder + '/binarized/' + filename)


for f in files(folder):
    binarizeImage(f)
