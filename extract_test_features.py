# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

#usage
# %run  ./Chapter03/extract_features.py --dataset ./datasets/animals/images --output ./datasets/animals/hdf5/features.hdf5
import sys
sys.path.append("..")

# import the necessary packages
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications.densenet import DenseNet201
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output HDFfile")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size in a convenience variable
bs = args["batch_size"]

# grab the list of images that we’ll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
#random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the
# labels
le = LabelEncoder()
M = open("datasets/RecTest/gt_id.txt").read().strip().split("\n")
M = [r.split()[:2] for r in M]
imagePaths = [os.path.sep.join(["datasets/m_tank_test", m[0]]) for m in M]
labels = le.fit_transform([m[1] for m in M])


# load the ResNet50 network
print("[INFO] loading network...")
#base_model = ResNet50(weights="imagenet", include_top=True)
#model = Model(inputs=base_model.input, 
#              outputs=base_model.get_layer("avg_pool").output)
base_model = DenseNet201(weights="imagenet", include_top=True)
model = Model(inputs=base_model.input, 
              outputs=base_model.get_layer("avg_pool").output)
# initialize the HDFdataset writer, then store the class label
# names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 1920),
                            args["output"], 
                            dataKey="features",
                            bufSize=args["buffer_size"])

dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    
    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224xpixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], 1920))

    # add the features and labels to our HDFdataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()

