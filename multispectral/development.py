import os
import subprocess
import shutil
import re
import numpy as np
import json
import math
import cv2
import tifffile
from tifffile import TiffFile, TiffWriter
import matplotlib.pyplot as plt
from cycler import cycler
from multispectral import Frame, Tools
from scipy import ndimage


class Flatfield:
    def __init__(self, reference_frame: Frame, filter_size=400):
        self.reference_frame = reference_frame
        self.filter_size = filter_size
        self.loaded_layer = None
        self.loaded_flatfield = None
        self.loaded_flatfield_max = 0

    def prepare(self, layer_type):
        """
        Loads the flatfield calibration image and blurs it
        :param layer_type: must match one of the layers in self.reference_frame
        :return:
        """
        matching_layers = [l for l in self.reference_frame.layers if l.name == layer_type]
        if len(matching_layers) == 1:
            self.loaded_layer = matching_layers[0]
            self.loaded_flatfield = tifffile.imread(self.loaded_layer.file)
            self.loaded_flatfield = cv2.blur(self.loaded_flatfield, (self.filter_size, self.filter_size))
            self.loaded_flatfield_max = np.max(self.loaded_flatfield[:])
        elif len(matching_layers) == 0:
            print('ERROR: no matching flatfield for %s' % layer_type)
        else:
            print('ERROR: multiple matching flatfields for %s' % layer_type)

    def correct(self, img_in):
        if img_in.shape == self.loaded_flatfield.shape:
            return np.divide(img_in, self.loaded_flatfield) * self.loaded_flatfield_max
        else:
            print('WARNING: flatfield and input image have different shapes! Returning original input image.')
            return img_in


class ReflectanceTarget:
    """
    Represents a reflectance target, including its location in the
    """
    FILE_COLORCHECKER_CALIB = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              'colorchecker-calibrated-intensities_kloburg2020.json')

    def __init__(self, center, radius, name, reflectance=-1.0):
        """
        """
        self.center = center
        self.radius = radius
        self.name = name
        self.reflectance = reflectance  # this can also be a dict
        self.intensities = {}

    def geom_to_int(self):
        self.center = (int(round(self.center[0])), int(round(self.center[1])))
        self.radius = int(round(self.radius))

    def to_dict(self):
        as_dict = {
            'center': self.center,
            'radius': self.radius,
            'name': self.name,
            'reflectance': self.reflectance,
            'intensities': self.intensities
        }
        return as_dict

    @staticmethod
    def from_dict(as_dict):
        t = ReflectanceTarget(
            center=as_dict['center'],
            radius=as_dict['radius'],
            name=as_dict['name'],
            reflectance=as_dict['reflectance']
        )
        t.intensities = as_dict['intensities']
        return t

    @staticmethod
    def name_to_reflectance(name: str, is_colorchecker: bool=False):
        '''
        :param name: string representing the reflectance value. can be a percentage (ending on %) or a ratio (just a number)
        :return: reflectance as ratio (float)
        '''
        if is_colorchecker:
            with open(ReflectanceTarget.FILE_COLORCHECKER_CALIB, 'r') as f:
                cc = json.load(f)
            try:
                targets = [t for t in cc['targets'] if t['name'] == name]
                intensities = targets[0]['intensities']
                for k in intensities.keys():
                    intensities[k] = intensities[k] / cc['max']
                return intensities
            except (KeyError, IndexError):
                print('could not convert string "%s" to reflectance' % name)
        else:
            name = name.replace(',', '.')
            if name.endswith("%"):
                is_percent = True
                name = name[:len(name) - 1]
            else:
                is_percent = False
            try:
                refl = float(name)
                if is_percent:
                    refl /= 100.0
                return refl
            except:
                print('could not convert string "%s" to reflectance' % name)


class ReflectanceCalibration:
    """
    Spectral Calibration based on reflectance targets
    """
    def __init__(self,
                 expected_dtype=None,
                 fallback_correction_fn=lambda x: x):
        """
        :param expected_dtype: you can provide the expected datatype of calibration- and
                                input images here. If none is given, the datatype is determined
                                by the first calibration image loaded.
        :param fallback_correction_fn: function if no correction function exists for the layer type
        """
        self.targets = []
        self.correction_fn = {}
        self.expected_dtype_max = None if expected_dtype is None else np.iinfo(expected_dtype).max
        self.fallback_correction_fn = fallback_correction_fn

    def read_target_definitions(self, labelme_json_file, is_colorchecker=False):
        """
        Reads the locations of targets in the images from a json file generated with the 'labelme' software
        :param labelme_json_file: a json file generated with 'labelme', containing circle labels on the targets.
        """
        f = open(labelme_json_file, 'r')
        obj = json.load(f)

        self.targets = []

        if 'shapes' in obj:
            for s in obj['shapes']:
                if s['shape_type'] == 'circle':
                    xm = s['points'][0][0]
                    ym = s['points'][0][1]
                    xp = s['points'][1][0]
                    yp = s['points'][1][1]
                    r = math.sqrt((xm - xp) ** 2 + (ym - yp) ** 2)
                    self.targets.append(ReflectanceTarget(
                        center=(xm, ym),
                        radius=r,
                        name=s['label'],
                        reflectance=ReflectanceTarget.name_to_reflectance(s['label'], is_colorchecker)
                    ))
                # add support for other shape types here

        if not self.targets:
            print('Error: no compatible target definitions found in ' + labelme_json_file)
            return False
        return True

    def read_target_intensities(self, frame: Frame, flatfield: Flatfield=None, plot=False):
        """
        Reads the average intensities of the target areas in the reference frame.
        Precondition: self.targets are populated with read_target_labels
        :param frame: a Frame which contains the targets
        :param flatfield: a Flatfield used to correct the target images prior to reading the values
        """
        #we work in pixel positions, so we cast those geometries to ints
        for t in self.targets:
            t.geom_to_int()

        # get means
        for layer in frame.layers:
            print('reading intensities for layer %s' % layer.name)
            img = tifffile.imread(layer.file)
            # remember the datatype of target images..
            if self.expected_dtype_max is None:
                self.expected_dtype_max = np.iinfo(img.dtype).max
            if flatfield is not None:
                flatfield.prepare(layer.name)
                img = flatfield.correct(img)
            for t in self.targets:
                #get mean intensity of target area
                values = []
                for row in range(t.center[1] - t.radius, t.center[1] + t.radius):
                    for col in range(t.center[0] - t.radius, t.center[0] + t.radius):
                        dx = col - t.center[0]
                        dy = row - t.center[1]
                        if math.sqrt(dx ** 2 + dy ** 2) < t.radius:
                            values.append(img[row, col])
                t.intensities[layer.name] = np.mean(values)

        return self.targets

    def plot_target_intensities(self, out_file=None):
        if not self.targets:
            print('nothing to plot')
            return

        layer_names = self.targets[0].intensities.keys()

        cm = plt.get_cmap('tab20')
        linestyle_cycler = cycler('color', [cm(i) for i in range(len(layer_names))])
        plt.rc('axes', prop_cycle=linestyle_cycler)
        for layer_name in layer_names:
            measured = []
            desired = []
            for t in self.targets:
                measured.append(t.intensities[layer_name])
                if isinstance(t.reflectance, dict):
                    desired.append(t.reflectance[layer_name])
                else:
                    desired.append(t.reflectance)
            measured.sort()
            plt.plot(desired, measured, label=layer_name[1:] if layer_name.startswith('_') else layer_name, marker='o')

        plt.title('Measured target intensities')
        plt.legend()
        if out_file is not None:
            plt.savefig(out_file)
        plt.show()

    def save_target_data(self, targets_file):
        with open(targets_file, 'w') as f:
            data = {'targets': [t.to_dict() for t in self.targets],
                    'max': self.expected_dtype_max}
            json.dump(data, f, indent=4)

    def load_target_data(self, targets_file):
        """
        Reads previously saved target information (including average intensities) from the given json file
        :param targets_file: a json file generated by save_target_data
        """
        try:
            with open(targets_file, 'r') as f:
                obj = json.load(f)
                self.expected_dtype_max = obj['max']
                self.targets = []
                for t in obj['targets']:
                    self.targets.append(
                        ReflectanceTarget.from_dict(t)
                    )
            return True
        except:
            return False

    def compute_correction_fn(self):
        layer_names = self.targets[0].intensities.keys()
        for layer_name in layer_names:
            # we expect linear sensor data at this point; so the correction can also be linear.
            # we fit a 1st degree fn v_tar = k*v_src + d, using the measured target values
            v_src = []  # measured values at the target areas
            v_tar = []  # expected reflectance of the targets
            for t in self.targets:
                if isinstance(t.reflectance, dict):
                    refl = t.reflectance[layer_name]
                else:
                    refl = t.reflectance
                v_src.append(t.intensities[layer_name])
                v_tar.append(refl * self.expected_dtype_max)

            c = np.polyfit(v_src, v_tar, 1)
            self.correction_fn[layer_name] = np.poly1d(c)


    def plot_correction_fn(self, out_file=None):
        if not self.correction_fn:
            print('nothing to plot')
            return

        layer_names = self.targets[0].intensities.keys()
        cm = plt.get_cmap('tab20')
        linestyle_cycler = cycler('color', [cm(i) for i in range(len(layer_names))])
        plt.rc('axes', prop_cycle=linestyle_cycler)
        for layer_name in layer_names:
            # we expect linear sensor data at this point; so the correction can also be linear.
            # we fit a 1st degree fn v_tar = k*v_src + d, using the measured target values
            v_src = []  # measured values at the target areas
            v_tar = []  # expected reflectance of the targets
            for t in self.targets:
                v_src.append(t.intensities[layer_name])
                if isinstance(t.reflectance, dict):
                    v_tar.append(t.reflectance[layer_name] * self.expected_dtype_max)
                else:
                    v_tar.append(t.reflectance * self.expected_dtype_max)


            plotx = np.arange(0, max(v_src) * 1.1)
            ploty = self.correction_fn[layer_name](plotx)
            plt.plot(plotx, ploty, label=layer_name[1:] if layer_name.startswith('_') else layer_name)
            plt.scatter(v_src, v_tar)

        plt.title('Correction curves')
        plt.legend()
        if out_file is not None:
            plt.savefig(out_file)
        plt.show()


    def correct(self, img_in, layer_type):
        """
        Applies the reflectance calibration
        :param img_in: input image
        :param layer_type: :param layer_type: a kex contained in self.correction_fn.keys()
        :return:
        """
        if layer_type in self.correction_fn.keys():
            img_corr = self.correction_fn[layer_type](img_in)
            img_corr = np.clip(img_corr, 0, self.expected_dtype_max)
            return img_corr
        else:
            print('WARNING: no matching correction function for %s. Applying fallback function.' % layer_type)
            return self.fallback_correction_fn(img_in)


class Development:
    @staticmethod
    def raw_to_linear_tiff(dir_in,
                           dir_out,
                           regex_in='IIQ$',
                           skip_existing=True):
        """
        converts all raw images in a directory to fully linear tiff images using dcraw: https://www.dechifro.org/dcraw/
        dcraw has to be installed on the machine and in the shell path in order for this to work
        :param dir_in: input directory
        :param dir_out: output directory
        :param dcraw_params: additional parameters for dcraw. default is '-4' --> linear 16 bit tiff
        :param regex_in: regular expression, determining input files to use
        :return:
        """

        print('<<<RAW development starting>>>')

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        files_to_convert = [f for f in os.listdir(dir_in) if re.search(regex_in, f)]
        i = 0
        for f in files_to_convert:
            file_in = os.path.join(dir_in, f)
            file_out_base = os.path.splitext(f)[0] + '.tiff'
            i += 1

            # check if already done
            if os.path.exists(os.path.join(dir_out, file_out_base)) and skip_existing:
                print('skipping %s' % file_in)
                continue

            # develop
            print('converting %s (%d/%d)...' % (file_in, i, len(files_to_convert)))
            subprocess.check_output(['dcraw',
                                     '-T',
                                     '-4',
                                     file_in])
            shutil.move(os.path.join(dir_in, file_out_base), os.path.join(dir_out, file_out_base))

        print('<<<RAW development completed>>>')

    @staticmethod
    def apply_adjustments(dir_in,
                          dir_out,
                          regex_in,
                          group_layername=0,
                          flatfield: Flatfield=None,
                          reflectance_cal: ReflectanceCalibration=None,
                          rotate=0,
                          rgb2gray=False,
                          medianfilter = 0,
                          new_dpi=None,
                          skip_existing=True):
        """
        Applies adjustments on the developed images, in the order of the function parameters
        :param dir_in: input directory
        :param dir_out: output directory
        :param regex_in: both a filter for input files and a hint to the
                        part of the filename determining the layer type
        :param group_layername: the group of regex_in determining the layer type. if set to 0 (default),
                                every file gets own group..
        :param flatfield: for flatfield calibration
        :param reflectance_cal: for reflectance calibration
        :param rotate: degrees of rotation (counter clockwise!); allowed values: 0, 90, 180, 270/-90
        :param rgb2gray: merge a 3 channel grayscale image to a single channel (via median)
        :param new_dpi: set the new dpi
        :param skip_existing: if true and the output file for a given input file already exists, skip
        :return:
        """

        print('<<<TIFF ADJUSTMENTS starting>>>')

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # for efficiency in flatfield correction we group the input files by layer type
        files_grouped = {}
        total_files = 0
        for f in os.listdir(dir_in):
            result = re.search(regex_in, f)
            if result:
                layer_name = result.group(group_layername)
                total_files += 1
                if layer_name in files_grouped.keys():
                    files_grouped[layer_name].append(f)
                else:
                    files_grouped[layer_name] = [f]

        i = 0
        for layer_name, files in files_grouped.items():
            # prepare flatfield correction
            if flatfield is not None:
                flatfield.prepare(layer_name)

            for f in files:
                file_in = os.path.join(dir_in, f)
                file_out = os.path.join(dir_out, f)
                i += 1

                # check if already done
                if os.path.exists(file_out) and skip_existing:
                    print('skipping %s' % file_in)
                    continue

                print('adjusting %s (%d/%d)...' % (file_in, i, total_files))

                # read tiff file
                tiff = TiffFile(file_in)
                img = tiff.pages[0].asarray()
                dtype_in = img.dtype

                # flatfield correction
                if flatfield is not None:
                    img = flatfield.correct(img)

                # reflectance calibration
                if reflectance_cal is not None:
                    img = reflectance_cal.correct(img, layer_name)

                # rotate
                if rotate == 0:
                    pass
                elif rotate == 90:
                    img = np.rot90(img, 1)
                elif rotate == 180:
                    img = np.rot90(img, 2)
                elif rotate == -90 or rotate == 270:
                    img = np.rot90(img, 3)
                else:
                    print("WARNING: invalid rotation angle provided (%d). Not rotating." % rotate)

                # rgb2gray
                if rgb2gray:
                    img = Tools.rgb2gray(img, Tools.Stat.MEDIAN)

                # apply medianfilter (e.g. for removing sensor noise)
                if medianfilter > 0:
                    img = ndimage.median_filter(img, (medianfilter*2+1, medianfilter*2+1))

                # set dpi
                if new_dpi is not None:
                    resolution = (new_dpi, new_dpi)
                else:
                    resolution = (tiff.pages[0].tags['XResolution'].value,
                                  tiff.pages[0].tags['YResolution'].value)

                # save image
                with TiffWriter(file_out) as tw:
                    tw.save(img.astype(dtype_in), resolution=resolution, compress=6)
                # copy all metadata from original, except for Resolution (because we just set that to the right value)
                Tools.copy_exif(file_in, file_out, omit_dpi=True)

        print('<<<TIFF ADJUSTMENTS completed>>>')
