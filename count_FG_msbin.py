import glob
import os
import sys
import cv2 as cv
import numpy as np

GTFolder = sys.argv[1]

def printEvaluation(fg1, fg2):
    all = fg1 + fg2

    print("FG1: " + str(fg1) + " (" + str((fg1 * 100)/all) + "%)")
    print("FG2: " + str(fg2) + " (" + str((fg2 * 100)/all) + "%)")

def calculateValues(gt):
    g = cv.imread(gt)

    fg2 = cv.inRange(g, np.array([121, 121, 121]), np.array([123, 123, 123]))
    fg1 = cv.inRange(g, np.array([254, 254, 254]), np.array([256, 256, 256]))

    #g = cv.resize(g, (1000, 500))
    #cv.imshow('ground truth', g)
    #cv.waitKey(0)

    #temp_fg1 = cv.resize(fg1, (1000, 500))
    #cv.imshow('FG1', temp_fg1)
    #cv.waitKey(0)

    #temp_fg2 = cv.resize(fg2, (1000, 500))
    #cv.imshow('FG2', temp_fg2)
    #cv.waitKey(0)


    fg1 = cv.countNonZero(fg1)
    fg2 = cv.countNonZero(fg2)

    return fg1, fg2

# counts the pixels for FG1 and FG2 in MSBin dataset and prints the results.
def countFG():
    GTImages = glob.glob(GTFolder + '/*.png')
    print("Found gt images: " + str(GTImages))

    fg1Total = 0
    fg2Total = 0

    for gt in GTImages:
        print("GT: " + os.path.basename(gt))

        fg1, fg2 = calculateValues(gt)

        fg1Total = fg1Total + fg1
        fg2Total = fg2Total + fg2

        printEvaluation(fg1, fg2)

    print("\n\nTotal Results")
    printEvaluation(fg1Total, fg2Total)

if __name__ == '__main__':
    countFG()
