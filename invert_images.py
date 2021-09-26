import sys
import os
from PIL import Image, ImageOps

# This file provides a function for inverting a image. (NOT FOR STANDALONE USE)
# Comments are here to enable export of the function.

#folder = sys.argv[1]
#outputFolder = sys.argv[2]

def invertImage(inputFilePath, outputFilePath):
    im = Image.open(inputFilePath)
    im_invert = ImageOps.invert(im)
    im_invert.save(outputFilePath)


#for f in os.listdir(folder):
#    invertImage(folder + '/' + f, outputFolder + '/' + f)
