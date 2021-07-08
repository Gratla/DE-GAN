import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
import pickle
from skimage.transform import resize, warp
from scipy.io import loadmat


#############################################
######### Geometric Transformations #########
#############################################
class Transformation(ABC):
    """
    Transformation base class.
    """

    @staticmethod
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def transform(self, img: np.array):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    def load(path):
        if os.path.splitext(path)[1] == '.mat':
            # expecting Matlab .mat file
            try:
                mat = loadmat(path)
                if 'flowX' in mat and 'flowY' in mat:
                    return TransformFlow(mat['flowX'], mat['flowY'])
                else:
                    raise
            except:
                print('WARNING: could not read transformation from file %s. Returning Identity.' % path)
                return TransformID()


        else:
            # expecting pickle file
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if 'type' in data:
                        if data['type'] == 'id':
                            return TransformID()
                        elif data['type'] == 'flow' and 'flow_x' in data and 'flow_y' in data:
                            return TransformFlow(data['flow_x'], data['flow_y'])
                        elif data['type'] == 'homo' and 'mat' in data:
                            return TransformHomo(data['mat'])
                        else:
                            raise
                    else:
                        raise
            except:
                print('WARNING: could not read transformation from file %s. Returning Identity.' % path)
                return TransformID()

class TransformID(Transformation):
    """
    Identity transformation (=no transformation)
    """
    @staticmethod
    def name():
        return "Identity"

    def transform(self, img: np.array):
        return img

    def save(self, path):
        data = {'type': 'id'}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

class TransformHomo(Transformation):
    """
    Homography transformation (represented by a 3x3 transformation matrix)
    """
    @staticmethod
    def name():
        return "Homography"

    def __init__(self, mat: np.array):
        self.mat = mat
    def transform(self, img: np.array):
        h, w = img.shape[:2]
        transformed = cv2.warpPerspective(img, self.mat, (w, h))
        return transformed
    def save(self, path):
        data = {'type': 'homo',
                'mat': self.mat}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

class TransformFlow(Transformation):
    """
    Flow transformation (represented by flow vector fields)
    """
    @staticmethod
    def name():
        return "Flow"

    def __init__(self, flow_x: np.array, flow_y: np.array):
        """
        :param flow_x: 2d matrix of horizontal flow component
        :param flow_y: 2d matrix of vertical flow component
        """
        self.flow_x = flow_x
        self.flow_y = flow_y

    def transform(self, img: np.array):
        nr, nc = img.shape[:2]
        # we allow flow low-res flow-fields; so we have to up-sample them to image resolution first
        flow_x = resize(self.flow_x, (nr, nc))
        flow_y = resize(self.flow_y, (nr, nc))
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        img_warp = warp(img.astype(np.float64), np.array([row_coords + flow_y, col_coords + flow_x]), mode='nearest')
        return img_warp

    def save(self, path):
        data = {'type': 'flow',
                'flow_x': self.flow_x,
                'flow_y': self.flow_y}
        with open(path, 'wb') as f:
            pickle.dump(data, f)



#############################################
######### Grayscale Transformations #########
#############################################
#todo