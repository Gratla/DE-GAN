import sys
import os
from PIL import Image, ImageOps


folder = sys.argv[1]

def scaleImage(filename):
    im = Image.open(folder + '/' + filename)
    im = im.convert('L')
    im = im.point(lambda x: 0 if x < 128 else 255, 'L')
    im.save(folder + '/../groundtruths_scaled_inverted/' + filename)


for f in os.listdir(folder):
    scaleImage(f)
