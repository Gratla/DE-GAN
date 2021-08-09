
#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from PIL import Image

input_size = (256,256,1)

def unet(pretrained_weights = None,input_size = input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate ([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate ([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate ([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate ([conv1,up9])  
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
    
    return model


def unet_bin(pretrained_weights = None,input_size = input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#     merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = concatenate ([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate ([conv3,up7])
#     merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate ([conv2,up8])
#     merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#     merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = concatenate ([conv1,up9])  
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
#     model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model




# The task defines the used weights for the generator. The string may have prefixes delimited by ';' to enable settings:
# 'T': Use threshold for enhanced image to binarize the DE-GAN output.
# 'S': Use half of the patch size as stride size. This will remove corruptions on the edges of the patches but will slow down the performance
mode = sys.argv[1].split(';')
task = mode.pop()

useThreshold = False
useHalfStride = False

for flag in mode:
    if flag == "T":
        useThreshold = True
    elif flag == "S":
        useHalfStride = True



if task =='binarize':
    generator = unet_bin()
    generator.load_weights("weights/binarization_generator_weights.h5")
else:
    if task == 'deblur':
        generator = unet_bin()
        generator.load_weights("weights/deblur_weights.h5")
    else:
        if task =='unwatermark':
            generator = unet()
            generator.load_weights("weights/watermark_rem_weights.h5")
        else:
            if task.startswith('epoch'):
                generator = unet_bin()
                generator.load_weights("weights/"+str(task)+"/weights/generator_weights.h5")
            else:
                print("Wrong task, please specify a correct task !")


def split2(dataset,size,h,w):
    newdataset=[]
    nsize1=256
    nsize2=256
    for i in range (size):
        im=dataset[i]
        for ii in range(0,h,nsize1):
            for iii in range(0,w,nsize2): 
                newdataset.append(im[ii:ii+nsize1,iii:iii+nsize2,:])
    
    return np.array(newdataset) 
def merge_image2(splitted_images, h,w):
    image=np.zeros(((h,w,1)))
    nsize1=256
    nsize2=256
    ind =0
    for ii in range(0,h,nsize1):
        for iii in range(0,w,nsize2):
            image[ii:ii+nsize1,iii:iii+nsize2,:]=splitted_images[ind]
            ind=ind+1
    return np.array(image)  


deg_image_path = sys.argv[2]

deg_image = Image.open(deg_image_path)# /255.0
deg_image = deg_image.convert('L')

test_image = np.divide(np.asfarray(deg_image.getdata()).reshape(deg_image.size[1], deg_image.size[0]), 255)


h =  ((test_image.shape [0] // 256) +1)*256 
w =  ((test_image.shape [1] // 256 ) +1)*256

test_padding=np.zeros((h,w))+1
test_padding[:test_image.shape[0],:test_image.shape[1]]=test_image

test_image_p=split2(test_padding.reshape(1,h,w,1),1,h,w)
predicted_list=[]
for l in range(test_image_p.shape[0]):
    predicted_list.append(generator.predict(test_image_p[l].reshape(1,256,256,1)))

predicted_image = np.array(predicted_list)#.reshape()
predicted_image=merge_image2(predicted_image,h,w)

predicted_image=predicted_image[:test_image.shape[0],:test_image.shape[1]]
predicted_image=predicted_image.reshape(predicted_image.shape[0],predicted_image.shape[1])
#     predicted_image = (predicted_image[:,:])*255

if useHalfStride:
    # Remove 128 pixel border to shift it from the original image
    shifted_test_padding = test_padding[128:test_padding.shape[0]-128, 128:test_padding.shape[1]-128].reshape(test_padding.shape[0]-256, test_padding.shape[1]-256)

    shifted_test_image_p = split2(shifted_test_padding.reshape(1, h-256, w-256, 1), 1, h-256, w-256)
    shifted_predicted_list = []
    for l in range(shifted_test_image_p.shape[0]):
        shifted_predicted_list.append(generator.predict(shifted_test_image_p[l].reshape(1, 256, 256, 1)))

    shifted_predicted_image = np.array(shifted_predicted_list)  # .reshape()
    shifted_predicted_image = merge_image2(shifted_predicted_image, h-256, w-256)

    shifted_predicted_image = shifted_predicted_image[:test_image.shape[0]-256, :test_image.shape[1]-256]
    shifted_predicted_image = shifted_predicted_image.reshape(shifted_predicted_image.shape[0], shifted_predicted_image.shape[1])

    # Use shifted prediction as mask
    predicted_image[128:-128, 128:-128][shifted_predicted_image > 0.95] = 1

if (task == 'binarize') or useThreshold:
    bin_thresh = 0.95
    predicted_image = (predicted_image[:,:]>bin_thresh)*1




save_path = sys.argv[3]

plt.imsave(save_path, predicted_image,cmap='gray')



