import os
import cv2
import re
import subprocess
import numpy as np
import tifffile
from multispectral import Transformation
from enum import Enum

class Rectangle:
    def __init__(self, top, left, rows, cols, round_to_int=False):
        self.top = top
        self.left = left
        self.rows = rows
        self.cols = cols

        if round_to_int:
            self.top = int(round(top))
            self.left = int(round(left))
            self.rows = int(round(rows))
            self.cols = int(round(cols))

class Tools:
    """Just some useful things."""

    #recursively transcodes a file tree
    @staticmethod
    def transcode(format, dir_in, dir_out, regex='.', enc_param=None):
        for d, _, files in os.walk(dir_in):
            for f in [f for f in files if re.search(regex, f, re.IGNORECASE)]:
                f_in = os.path.join(d, f)
                im = cv2.imread(f_in)
                if im is None:
                    continue

                basename = os.path.splitext(f)[0]
                format = format.replace('.', '')
                d_out = d.replace(dir_in, dir_out)
                if not os.path.exists(d_out):
                    os.makedirs(d_out)
                f_out = os.path.join(d_out, basename+'.'+format)

                if format.lower() in ['jpg', 'jpeg']:
                    if enc_param is None:
                        enc_param = 95
                    cv2.imwrite(f_out, im, (cv2.IMWRITE_JPEG_QUALITY, enc_param))
                elif format.lower() == 'png':
                    if enc_param is None:
                        enc_param = 3
                    cv2.imwrite(f_out, im, (cv2.IMWRITE_PNG_COMPRESSION, enc_param))

                print('Transcoded: %s' % f_out)

    @staticmethod
    def add_suffix(file, suffix):
        file_base, file_ext = os.path.splitext(file)
        return file_base + suffix + file_ext

    @staticmethod
    def crop_img(img, rect: Rectangle):
        return img[rect.top : rect.top+rect.rows,
                   rect.left : rect.left+rect.cols]

    @staticmethod
    def clean_path(path):
        path = path.replace('\\', os.path.sep).replace('/', os.path.sep)
        return path

    @staticmethod
    def copy_exif(file_src, file_dst, omit_dpi=False):
        """
        Copies complete exif data from file_src to file_dst, using exiftool: https://exiftool.org/exiftool_pod.html
        exiftool has to be installed on the machine and in the shell path for this to work
        :param file_src: take exif from this file
        :param file_dst: write exif to this file
        :param omit_dpi: special flag to omit copying of the dpi (e.g. if we have just set the correct dpi on a file, we don't want to overwrite it...
        :return:
        """
        # cmd = 'exiftool -overwrite_original -tagsFromFile %s%s%s' % (Tools.clean_path(file_src),
        #                                                             ' --XResolution --YResolution ' if omit_dpi else ' ',
        #                                                             Tools.clean_path(file_dst))
        cmd = ['exiftool',
                '-overwrite_original',
                '-tagsFromFile',
                Tools.clean_path(file_src)]
        if omit_dpi:
            cmd.append('--XResolution')
            cmd.append('--YResolution')
        cmd.append(Tools.clean_path(file_dst))

        subprocess.check_output(cmd)


    class Stat(Enum):
        MEAN = 0
        MEDIAN = 1

    @staticmethod
    def rgb2gray(img, method=Stat.MEAN):
        if len(img.shape) >= 3:
            if method == Tools.Stat.MEAN:
                return np.mean(img, 2)
            if method == Tools.Stat.MEDIAN:
                return np.median(img, 2)
        return img


    @staticmethod
    def load_img(file_img, to_gray: Stat=None, file_transform=None) -> np.array:
        print('Loading %s...' % file_img)
        if os.path.splitext(file_img.lower())[1] in ['.tiff', '.tif']:
            img = tifffile.imread(file_img)
        else:
            img = cv2.imread(file_img, cv2.IMREAD_UNCHANGED)

        if to_gray is not None:
            img = Tools.rgb2gray(img, to_gray)
        if file_transform is not None:
            trans = Transformation.load(file_transform)
            print('Applying %s transformation.' % trans.name())
            img = trans.transform(img)
        return img


    @staticmethod
    def save_img(img: np.array, file, force16bit=False, to_gray: Stat=None):
        print('Saving %s...' % file)

        if to_gray is not None:
            img = Tools.rgb2gray(img, to_gray)

        if os.path.splitext(file.lower())[1] in ['.tiff', '.tif']:
            # tifffile expects uint8 or uint16
            if img.max() >= 2**8 or force16bit:
                tifffile.imwrite(file, img.astype(np.uint16))
            elif img.max() > 1:
                tifffile.imwrite(file, img.astype(np.uint8))
            else:
                if force16bit:
                    tifffile.imwrite(file, (img * (2 ** 16 - 1)).astype(np.uint16))
                else:
                    tifffile.imwrite(file, (img * (2 ** 8 - 1)).astype(np.uint8))
        else:
            cv2.imwrite(file, img)


    @staticmethod
    def normalize(img: np.array, cutoff=0.001):
        """
        min-max normalization of an image
        :param img: input image
        :param cutoff: for determining min and max, throw away the given fraction of highest and lowest values
                        (to deal with outliers and pixel errors)
        :return: normalized image
        """
        n_remove = int(img.size * cutoff)
        vals = img.reshape(img.size)
        vals = np.partition(vals, n_remove)
        min = vals[n_remove]
        vals = np.partition(vals, vals.size-n_remove)
        max = vals[vals.size-n_remove]


        if max < 1:
            # in [0,1]
            newmax = 1
        elif max < 2**8:
            # 8 bit
            newmax = 2**8-1
        else:
            # 16 bit
            newmax = 2**16-1

        # normalize
        img = img.astype(np.float64) - min
        img = img * (newmax / max)
        # clip
        img[img < 0] = 0
        img[img > newmax] = newmax

        return img