#TODO:
# - dynamically chose the number of PCA components by specifying the "variance explained" (e.g. 95%). Is that possible
#   for other methods?
# super nice resource: https://www.bogotobogo.com/python/scikit-learn/scikit_machine_learning_Data_Compresssion_via_Dimensionality_Reduction_1_Principal_component_analysis%20_PCA.php

import logging
import os
import cv2
import numpy as np
from time import time
from enum import Enum
from numpy.random import RandomState
from sklearn import cluster, decomposition, preprocessing
from multispectral import Layer, Frame, Tools


class Unmixing:
    """
    Performs several decomposition algorithms or k-means clustering (from scikit.learn) on a Frame,
    where a pixel positions correspond to observations and Layers correspond to features.
    """

    class Method(Enum):
        #decomposition
        PCA = 1     #principal component analysis
        ICA = 2     #independent component analysis
        FA = 3      #factor analysis
        TSVD = 4    #truncated singular value decomposition
        NMF = 5     #non-negative matrix factorization
        #clustering
        KMEANS = 6  #k-means clustering

    class Implementation:
        def __init__(self, name, estimator):
            self.name = name
            self.estimator = estimator


    def __init__(self, frame, convert_to_grayscale=False):
        """
        :param frame: Frame that will be processed. Has to be provided upon construction, because the actual pixel data
                        and metadata are loaded here.
        :param convert_to_grayscale: if true, loaded rgb images will be converted to grayscale.
                        otherwise, 3 layers are extracted from an rgb image.
        """
        self.rng = RandomState(0)
        self.frame = frame
        self.image_shape = [0, 0]
        self.source_data = None
        self.n_features = 0
        self.n_samples = 0

        self.__load_images(convert_to_grayscale)


    def __load_images(self, convert_to_grayscale, center=True):
        """
        Reads image files as gray value images.
        :param center: center the data around zero?
        :return:
        """
        data_list = []
        for i, layer in enumerate(self.frame.layers):
            img = Tools.load_img(file_img=layer.file,
                                  to_gray=Tools.Stat.MEAN if convert_to_grayscale else None,
                                  file_transform=layer.transform_file)

            # crop image. for unmixing, image should be cropped to the area with interesting content
            if self.frame.crop is not None:
                img = Tools.crop_img(img, self.frame.crop)

            # if this is the first image that was loaded, we store the images width and height (for reconstruction..)
            if i < 1:
                self.image_shape = img.shape[0:2]

            # reshape image to 1d vector and append to data list
            if len(img.shape) > 2:  #multi-channel image will be split up
                for c in range(img.shape[2]):
                    s = int(img.size/img.shape[2])
                    img_reshaped = np.reshape(img[:, :, c], s)
                    data_list.append(img_reshaped)
            else:
                data_list.append(np.reshape(img, img.size))


        # convert to 2d np array, prepare for unmixing
        data_array = np.asarray(data_list)
        data_array = np.transpose(data_array)
        self.n_samples, self.n_features = data_array.shape

        if center:
            # global centering
            self.source_data = data_array - data_array.mean(axis=0)
            # local centering
            self.source_data -= self.source_data.mean(axis=1).reshape(self.n_samples, -1)
        else:
            self.source_data = data_array

        print("Frame %s consists of %d images" % (self.frame.name, self.n_features))


    def __get_implementation(self, method:Method, n_components):
        # decomposition
        if method == self.Method.PCA:
            return self.Implementation('pca', decomposition.PCA(n_components=n_components, svd_solver='full', whiten=True))
        elif method == self.Method.ICA:
            return self.Implementation('ica', decomposition.FastICA(n_components=n_components, whiten=True, max_iter=1000))
        elif method == self.Method.FA:
            return self.Implementation('fa', decomposition.FactorAnalysis(n_components=n_components))
        elif method == self.Method.TSVD:
            return self.Implementation('tsvd', decomposition.TruncatedSVD(n_components=n_components))
        elif method == self.Method.NMF:
            return self.Implementation('nmf', decomposition.NMF(n_components=n_components))
        #clustering
        elif method == self.Method.KMEANS:
            return self.Implementation('kmeans', cluster.MiniBatchKMeans(n_clusters=n_components, tol=1e-3))
        else:
            raise Exception('Error creating estimator. Invalid Type specified.')


    def unmix(self, method:Method, n_components=0, out_dir='', out_extension='tif', p_keep=1.0, verbose=False):
        """
        Does the thing.
        :param method: Choose your weapon.
        :param n_components: How many output components do you want?
        :param out_dir: Where to put the resulting images? Defaults to: os.path.join(self.frame.root_dir, method_name)
        :param out_ext: Image file extension for output. Defaults to tif.
        :param p_keep: For large images, the decomposution/clustering can be computed using a random subset of the
                    samples. Set to a value between 0 and 1 to set the fraction of total data that should be used.
        :param verbose: if true, images are shown with matplotlib upon storing
        :return: A Frame containing the visualizations of the results.
        """
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        #if n_components undefined, use number of input features
        if n_components < 1:
            n_components = self.n_features

        imp = self.__get_implementation(method, n_components)

        print("Extracting %d components using %s..." % (n_components, imp.name))
        t0 = time()

        if p_keep >= 1.0:
            fitting_data = self.source_data
        else:
            # select random subset of samples (probably one could do this more elegnatly)
            idx = np.arange(self.n_samples)
            np.random.shuffle(idx)
            idx = idx[:int(self.n_samples * p_keep)]
            fitting_data = self.source_data[idx, :]

        if imp.name=='nmf':
            #special case: does not support negative values
            imp.estimator.fit(fitting_data + np.ones(np.shape(fitting_data)) * -np.min(fitting_data))
        else:
            imp.estimator.fit(fitting_data)

        train_time = (time() - t0)
        print("done in %0.3fs" % train_time)

        if imp.name=='nmf':
            result_data = imp.estimator.transform(self.source_data +
                                                  np.ones(np.shape(self.source_data)) * -np.min(self.source_data))
        else:
            result_data = imp.estimator.transform(self.source_data)

        return self.__store_results(imp.name, result_data, out_dir, out_extension, verbose)



    def __store_results(self, method_name, images, out_dir, out_ext, verbose):
        if out_dir == '':
            #subfolder of input dir
            out_dir = os.path.join(self.frame.root_dir, method_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        images = np.transpose(images)

        frame_out = Frame(name='%s_%s' % (self.frame.name, method_name),
                          root_dir=out_dir, parse_root_dir=False)

        for i, comp in enumerate(images):
            vmax = max(comp.max(), -comp.min())

            if verbose:
                import matplotlib.pyplot as plt
                plt.imshow(comp.reshape(self.image_shape), cmap=plt.cm.gray,
                           interpolation='nearest',
                           vmin=-vmax, vmax=vmax)

            out_file = '%s_%s%s.%s' % (self.frame.name, method_name, str(i).zfill(2), out_ext)
            out_fullfile = os.path.join(out_dir, out_file)
            out_img = np.uint8(preprocessing.minmax_scale(comp, feature_range=(0, 255)).reshape(self.image_shape))
            cv2.imwrite(out_fullfile, out_img)
            frame_out.append(Layer(name=os.path.splitext(out_file)[0],
                                   file=out_fullfile))

        return frame_out