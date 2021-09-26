import sys
import os
from PIL import Image
from numpy import asarray

# This script converts all images in the passed folder into an 8-bit greyscale image and stores them into a subfolder
# called "converted".
# usage: py convert_images.py ./your/folder/path

folder = sys.argv[1]

def convertImage(filename):
    im = Image.open(folder + '/' + filename)
    im = asarray(im)
    im = im.astype('float64')
    im *= 255.0 / im.max()
    im = Image.fromarray(im)
    im = im.convert('L')
    im.save(folder + '/converted/' + filename)

    # clean_image_path = ('data/B/' + list_clean_images[im])
    # clean_image = Image.open(clean_image_path)  # /255.0
    # clean_image = clean_image.convert('L')
    # clean_image.save('curr_clean_image.png')


for f in os.listdir(folder):
    convertImage(f)
