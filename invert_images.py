import sys
import os
from PIL import Image, ImageOps


folder = sys.argv[1]
outputFolder = sys.argv[2]

def invertImage(filename):
    im = Image.open(folder + '/' + filename)
    im_invert = ImageOps.invert(im)
    im_invert.save(outputFolder + '/' + filename)


for f in os.listdir(folder):
    invertImage(f)
