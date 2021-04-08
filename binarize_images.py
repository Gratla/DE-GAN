import sys
import os
from PIL import Image, ImageOps


folder = sys.argv[1]

def binarizeImage(filename):
    im = Image.open(folder + '/' + filename)
    im = im.convert('L')
    im = im.point(lambda x: 0 if x < 128 else 255, '1')
    im.save(folder + '/../groundtruths_binarized/' + filename)


for f in os.listdir(folder):
    binarizeImage(f)
