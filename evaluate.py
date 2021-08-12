import glob
import sys
import cv2 as cv
import numpy as np

enhancedFolder = sys.argv[1]
GTFolder = sys.argv[2]

def calculateValues(enhanced, gt):
    e = cv.imread(enhanced)
    e = cv.bitwise_not(e)
    g = cv.imread(gt)

    notUsedPixelMask = cv.inRange(g, np.array([254, 0, 0]), np.array([255, 0, 0]))
    notUsedPixelMask = cv.bitwise_not(notUsedPixelMask)

    fg2 = cv.inRange(g, np.array([121, 121, 121]), np.array([123, 123, 123]))
    g[fg2==255] = [255, 255, 255]

    e = cv.cvtColor(e, cv.COLOR_BGR2GRAY)
    g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)

    e = cv.bitwise_and(e, notUsedPixelMask)
    g = cv.bitwise_and(g, notUsedPixelMask)

    tp = cv.bitwise_and(e, g)
    #tp_1 = cv.resize(tp, (1000, 500))
    #cv.imshow('True Positives', tp_1)
    #cv.waitKey(0)

    tn = cv.bitwise_and(cv.bitwise_not(e), cv.bitwise_not(g))
    #tn = cv.resize(tn, (1000, 500))
    #cv.imshow('True Negatives', tn)
    #cv.waitKey(0)

    fp = cv.bitwise_xor(e, tp)
    #fp = cv.resize(fp, (1000, 500))
    #cv.imshow('False Positives', fp)
    #cv.waitKey(0)

    fn = cv.bitwise_xor(g, tp)
    #fn = cv.resize(fn, (1000, 500))
    #cv.imshow('False Negatives', fn)
    #cv.waitKey(0)

    #e = cv.resize(e, (1000, 500))
    #cv.imshow('enhanced', e)
    #cv.waitKey(0)

    #g = cv.resize(g, (1000, 500))
    #cv.imshow('ground truth', g)
    #cv.waitKey(0)

    tp = cv.countNonZero(tp)
    tn = cv.countNonZero(tn)
    fp = cv.countNonZero(fp)
    fn = cv.countNonZero(fn)

    recall = tp/(tp + fn)
    precision = tp/(tp + fp)

    fm = (2*recall*precision)/(recall+precision)


    nrfn = fn/(tp+fn)
    nrfp = fp/(fp+tn)
    nrm = (nrfn+nrfp)/2

    return fm, nrm


# evaluates the f-measure and NRM for the given input images
def evaluate():
    enhancedImages = glob.glob(enhancedFolder + '/*.png')
    GTImages = glob.glob(GTFolder + '/*.png')
    print("Found enhanced images: " + str(enhancedImages))
    print("Found gt images: " + str(GTImages))

    assert len(enhancedImages) == len(GTImages)

    for enhanced, gt in zip(enhancedImages, GTImages):
        print(enhanced + " - " + gt + ":")

        f, n = calculateValues(enhanced, gt)

        print("f-measure: " + str(f))
        print("NRM: " + str(n))



if __name__ == '__main__':
    evaluate()
