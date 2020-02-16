#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:52:03 2019

@author: sunw71
"""

import sys
sys.path.append("..")

# %run ./dogs_vs_cats/train_alexnet.py
# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
#from config import dogs_vs_cats_config as config
from config import tank as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2, 
                         shear_range=0.15,
                         horizontal_flip=True, 
                         fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
#pp = PatchPreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 
                                32,
                                aug=aug,
                                preprocessors=[sp, mp, iap],
                                classes=205)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, 
                              32,
                              preprocessors=[sp, mp, iap], 
                              classes=205)


# initialize the optimizer
print("[INFO] compiling model...")
#opt = Adam(lr=1e-3)
opt=SGD(lr=0.1)
model = AlexNet.build(width=227, 
                      height=227, 
                      depth=3,
                      classes=205, 
                      reg=0.0002)

model.compile(loss="categorical_crossentropy", 
              optimizer=opt,
              metrics=["accuracy"])


# construct the set of callbacksm1
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])

callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages // 32,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // 32,
                    epochs=50,
                    max_queue_size=32 * 2,
                    callbacks=callbacks, 
                    verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()