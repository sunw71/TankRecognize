#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:32:50 2019

@author: sunw71
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.append("..")
# import the necessary packages
from config import tank as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", default="output/checkpoints/",
                 help="path to output checkpoint directory")

ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")

ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")



args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, 
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True, 
                         fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, 
                                aug=aug,preprocessors=[sp, mp, iap], 
                                classes=config.NUM_CLASSES)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32,
                              preprocessors=[sp, mp, iap], 
                              classes=config.NUM_CLASSES)


# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    model = DeeperGoogLeNet.build(width=224, height=224, depth=3,
                                  classes=config.NUM_CLASSES,
                                  reg=0.0002)

#    opt = Adam(1e-3)
    opt = SGD(lr=1e-2,momentum=0.9)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,metrics=["accuracy"])


# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))

    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


# construct the set of callbacks
callbacks = [
    EpochCheckpoint(args["checkpoints"],
                     every=5,
                     startAt=args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, 
                    jsonPath=config.JSON_PATH,
                    startAt=args["start_epoch"])]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 32,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 32,
    epochs=30,
    max_queue_size=32 * 2,
    callbacks=callbacks, verbose=1)

# close the databases
trainGen.close()
valGen.close()
