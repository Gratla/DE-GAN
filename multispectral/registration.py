import os
import re
from multispectral import Frame, Tools, Transformation, TransformFlow, TransformID, TransformHomo
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.registration import optical_flow_tvl1
from abc import ABC, abstractmethod
import time, datetime

# registration resources:
# list of deformable registration projects: http://pyimreg.github.io/
# MIND & co: http://www.mpheinrich.de/software.html


###########################################
######### Registration Algorithms #########
###########################################
class RegAlg(ABC):
    @abstractmethod
    def register(self, img_fixed, img_moving) -> Transformation:
        pass

# FAIL
class RegAlgSkimg(RegAlg):
    def register(self, img_fixed, img_moving):
        flow_y, flow_x = optical_flow_tvl1(img_fixed, img_moving)
        return TransformFlow(flow_x, flow_y)

# SUCCESS for fine registration
class RegAlgMind(RegAlg):

    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def register(self, img_fixed, img_moving):
        import matlab
        import mind

        client = mind.initialize()

        img_fixed_ml = matlab.double(img_fixed.tolist())
        img_moving_ml = matlab.double(img_moving.tolist())
        fx, fy, warped = client.deformableReg2Dmind(img_fixed_ml, img_moving_ml, self.alpha, nargout=3)

        return TransformFlow(np.array(fx, dtype=np.single), np.array(fy, dtype=np.single))


# PARTIAL FAIL for fine registration, for certain images.
# (could work perfectly well for registering P1-white and D4-white images, though)
class RegAlgOrb(RegAlg):
    def register(self, img_fixed, img_moving):

        # scale to [0, 255]
        img_fixed_cv = img_fixed / np.max(img_fixed[:]) * 255
        img_moving_cv = img_moving / np.max(img_moving[:]) * 255

        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(5000)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        #  (which is not reqiured in this case).
        kp_moving, d_moving = orb_detector.detectAndCompute(img_moving_cv.astype(np.uint8), None)
        kp_fixed, d_fixed = orb_detector.detectAndCompute(img_fixed_cv.astype(np.uint8), None)

        # Match features between the two images.
        # We create a Brute Force matcher with
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(d_moving, d_fixed)

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches) * 90)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p_moving = np.zeros((no_of_matches, 2))
        p_fixed = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p_moving[i, :] = kp_moving[matches[i].queryIdx].pt
            p_fixed[i, :] = kp_fixed[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p_moving, p_fixed, cv2.RANSAC)

        return TransformHomo(homography)


##############################################
######### The Registration Framework #########
##############################################
class Registration:
    """ Contains functions for parametric feature based and non-parametric (deformable) registration"""

    @staticmethod
    def determine_ref_layer(frame: Frame, regex_ref):
        ref_layer = None

        if len(frame.layers) == 0:
            print('ERROR: empty frame provided.')
            return

        ref_layers = [l for l in frame.layers if re.search(regex_ref, l.name)]
        if len(ref_layers) == 0:
            ref_layer = frame.layers[0]
            print('WARNING: no layer matches %s. Using fallback reference: first layer of the frame (%s).' %
                  (regex_ref, ref_layer.name))
        elif len(ref_layers) == 1:
            ref_layer = ref_layers[0]
            print('Using layer %s as reference.' % ref_layer.name)
        else:
            ref_layer = ref_layers[0]
            print('WARNING: multiple layers match %s. Using first one (%s).' % (regex_ref, ref_layer.name))

        return ref_layer


    @staticmethod
    def register(frame: Frame,
                regex_ref,
                algorithm: RegAlg,
                dir_out,
                regex_skip=None,
                skip_existing=True,
                visualize=False):

        """
        Registers all layers of frame to the layer determined by regex_refr.
        Stores the resulting Transformations.
        :param frame: frame to be registered. resulting transformations are added to layers.
        :param regex_ref: regular expression to determine the reference image
        :param algorithm: multiple registration algorithms are available
        :param dir_out: directory to which transformations should be saved
        :param regex_skip: provide a fegex matching files you would not like to register (e.g. because they already match well)
        :param skip_existing: if true and the output file already exists, skips this.
        :param visualize: visualize computed transformations?
        """

        # initialization stuff
        ref_layer = Registration.determine_ref_layer(frame, regex_ref)

        # load reference image
        img_fixed = Tools.load_img(ref_layer.file, to_gray=Tools.Stat.MEAN)

        # make output directory and put reference image there
        if not os.path.isabs(dir_out):
            dir_out = os.path.join(frame.root_dir, dir_out)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        for layer in [la for la in frame.layers
                      if la.name != ref_layer.name
                         and not (re.search(regex_skip, la.file) if regex_skip else False)]:

            file_out = os.path.join(dir_out, os.path.basename(layer.file) + '.transf')
            if os.path.exists(file_out) and skip_existing:
                print('skipping %s because transform already exists.' % layer.file)
                continue

            print('registering %s...' % layer.file)
            t = time.time()
            img_moving = Tools.load_img(layer.file, to_gray=Tools.Stat.MEAN)

            transform = algorithm.register(img_fixed, img_moving)

            # visualization (optional)
            if visualize:
                if isinstance(transform, TransformFlow):
                    plt.imshow(transform.flow_x)
                    plt.colorbar()
                    plt.title('%s: x-flow' % layer.name)
                    plt.show()
                    plt.imshow(transform.flow_y)
                    plt.colorbar()
                    plt.title('%s: y-flow' % layer.name)
                    plt.show()

            layer.transform_file = file_out
            print('Registered in %s seconds. Saving transformation to %s.' %
                  (str(datetime.timedelta(seconds=time.time() - t)), file_out))
            transform.save(layer.transform_file)


    @staticmethod
    def visualize_errors(frame: Frame, regex_ref, output_dir=None):
        ref_layer = Registration.determine_ref_layer(frame, regex_ref)

        ref_img = cv2.imread(ref_layer.file, cv2.IMREAD_GRAYSCALE)
        for layer in [la for la in frame.layers if la.name != ref_layer.name]:
            moving_img = cv2.imread(layer.file, cv2.IMREAD_GRAYSCALE)
            # apply transformation if exists
            if layer.transform_file is not None:
                transf = Transformation.load(layer.transform_file)
                moving_img = transf.transform(moving_img)

            # crop moving image if too lage now
            if moving_img.shape[0] > ref_img.shape[0]:
                moving_img = moving_img[:ref_img.shape[0], :]
            if moving_img.shape[1] > ref_img.shape[1]:
                moving_img = moving_img[:, :ref_img.shape[1]]
            # padd with zeros if it is too small now
            moving_img_padded = np.zeros(ref_img.shape).astype(np.uint8)
            moving_img_padded[:moving_img.shape[0], :moving_img.shape[1]] = moving_img

            vis_img = np.dstack([ref_img, moving_img_padded, moving_img_padded])
            if output_dir is None:
                plt.imshow(vis_img)
                plt.show()
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                outfile = os.path.join(output_dir, os.path.basename(layer.file))
                Tools.save_img(vis_img, outfile)





    # def register_sitk(self, img_fixed: np.array, img_moving: np.array, ttype='bspline', dir_out='registered', suffix_out='_freg', verbose=False):
    #     """
    #     Registers two images using SimpleElastix. This function needs elastix binaries installed on your machine.
    #     http://elastix.isi.uu.nl/
    #     :param ttype: transformation type. translation, affine, bspline are valid options.
    #     :param dir_out: output directory for registered images. Can be absolute or relative (to frame.root_dir)
    #     :param suffix_out: suffix that will be attached to original filename to indicate its registration status
    #     :param verbose: if True, registration success is visualized. only for testing (waits for input after each image) (TODO)
    #     :return: Frame of now registered images
    #     """
    #     import SimpleITK as sitk
    #
    #     #find and load reference layer
    #     ref_img_sitk = sitk.GetImageFromArray(img_fixed)
    #     #ref_img_sitk = sitk.ReadImage(ref_layer.file)
    #     #ref_img_sitk.SetSpacing((1.0, 1.0))     #safety
    #
    #     moving_img_sitk = sitk.GetImageFromArray(img_moving)
    #     # moving_img_sitk = sitk.ReadImage(layer.file)
    #     # moving_img_sitk.SetSpacing((1.0, 1.0))
    #
    #     # create registration object and add image pair
    #     filter = sitk.ElastixImageFilter()
    #     filter.SetFixedImage(ref_img_sitk)
    #     filter.SetMovingImage(moving_img_sitk)
    #
    #     # define parameters
    #     params = sitk.GetDefaultParameterMap(ttype)
    #     if ttype == "bspline":
    #         params['MaximumNumberOfIterations'] = ['512']
    #         params['FinalGridSpacingInPhysicalUnits'] = ['100']
    #         # params_b['FinalGridSpacingInPhysicalUnits'] = ['300']
    #         # TODO: make this non-hardcoded, or dependent of resolution..
    #
    #     filter.SetParameterMap(params)
    #     filter.PrintParameterMap()
    #     # TODO: as we apply the registration to only one image here, we could also use the default multi-resolution approach here..
    #     # ("ElastixImageFilter will register our images with a translation -> affine -> b-spline multi-resolution approach by default.")
    #
    #     # do registration
    #     filter.Execute()
    #
    #     # save registered image
    #     result_img_sitk = filter.GetResultImage()
    #
    #     # sitk.WriteImage(result_img_sitk, file_out)  #somehow this doesn't produce proper images..
    #     result_img = sitk.GetArrayFromImage(result_img_sitk)
    #     if np.max(result_img) >= 2 ** 8:
    #         result_img = np.uint16(result_img)
    #     else:
    #         result_img = np.uint8(result_img)
    #
    #     return result_img
