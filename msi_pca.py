import os

import cv2 as cv

from multispectral import Frame, Unmixing
import numpy as np
from scipy.signal import argrelextrema


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def executePCA(msiName, msiPath, outputPath, n_components):
    # collect images in root_dir matching regex; groups 1 and 2 of the match object
    # identify the document and the layer respectively (optional)
    frame = Frame(root_dir=msiPath,
                  regex='(' + msiName + ')_F(\d+)s.png',
                  group_framename=1,
                  group_layername=2)

    # make unmixing object: loads images of frame and converts them to a data matrix
    um = Unmixing(frame)
    # perform principal component analysis, store visualizations of first 5 components
    # in given output folder (or by default frame.root_dir/pca), return frame containing those
    principal_components = um.unmix(method=Unmixing.Method.PCA, n_components=n_components, out_dir=outputPath,
                                    out_extension='png',
                                    verbose=True)

def invertIfNeed(folder):
    cntNonInv = 0
    for f in files(folder):
        im = cv.imread(folder + "/" + f)

        histogram = np.histogram(im, bins=np.arange(256))
        maxima = argrelextrema(histogram[0], np.greater, order=1)

        if (f.endswith("EA67_pca00.png")):
            print("maxima EA67:")
            print(str(maxima))

        if (f.endswith("EA0_pca00.png")):
            print("maxima EA0:")
            print(str(maxima))

        if len(maxima) >= 2:
            first = 0
            second = 0

            for i in range(0, len(maxima)):
                if i < len(maxima)//2:
                    first = first + maxima[i]
                else:
                    second = second + maxima[i]


            if first > second:
                im_invert = np.iinfo(im.dtype).max - im
                cv.imwrite(folder + "/pcaInverted/" + f, im_invert)
            else:
                cv.imwrite(folder + "/pcaInverted/" + f, im)
                cntNonInv = cntNonInv + 1

        else:
            avg_color_per_row = np.average(im, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)

            temp = im.copy()
            temp[temp < avg_color[0]] = 0
            temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)

            if cv.countNonZero(temp) < ((temp.shape[0]*temp.shape[1])//2):
                #print(str(cv.countNonZero(temp)) + "<" + str(int(((temp.shape[0]*temp.shape[1])//2))))
                im_invert = np.iinfo(im.dtype).max - im
                cv.imwrite(folder + "/pcaInverted/" + f, im_invert)
            else:
                cv.imwrite(folder + "/pcaInverted/" + f, im)
                cntNonInv = cntNonInv + 1

            if (f.endswith("BT17_pca00.png1")):
                temp = cv.resize(temp, (1000, 500))
                cv.imshow('test', np.concatenate(temp, np.histogram2d(temp)))
                cv.waitKey(0)

    print(str(cntNonInv))