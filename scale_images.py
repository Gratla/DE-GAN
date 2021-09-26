import sys
import os
from PIL import Image, ImageOps

# This script scales pixel values of the images provided in the passed folder.
# It uses a static threshold for binarizing the images.
# This script is not used in the thesis!
#
# usage: py scale_images.py ./your/folder/path

folder = sys.argv[1]

def scaleImage(filename):
    im = Image.open(folder + '/' + filename)
    im = im.convert('L')
    im = im.point(lambda x: 0 if x < 128 else 255, 'L')
    im.save(folder + '/../groundtruths_scaled_inverted/' + filename)


for f in os.listdir(folder):
    scaleImage(f)
