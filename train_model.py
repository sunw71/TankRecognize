# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", default='datasets/m_tank_train/hdf5/densnet_features121.hdf5',
                help="path HDF5 database")
ap.add_argument("-m", "--model", default='output/DensenetLR.pickle',
                help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(args["db"], "r")
#i = int(db["labels"].shape[0] * 0.75)
db_val = h5py.File('datasets/m_tank_val/hdf5/densnet_features121.hdf5', "r")

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
#params = {"C": [1.0]}
#model = GridSearchCV(LogisticRegression(multi_class="multinomial",solver="lbfgs"), 
#                     params,
#                     cv=3,
#                     n_jobs=args["jobs"])
model = SVC(kernel="rbf", gamma=5, C=0.001,probability=True)
model.fit(db["features"][:], db["labels"][:])
#print("[INFO] best hyperparameters: {}".format(model.best_params_))

# generate a classification report for the model
print("[INFO] evaluating...")
preds = model.predict(db_val["features"])
print(classification_report(db_val["labels"], 
                            preds,
                            target_names=db_val["label_names"]))

# compute the raw accuracy with extra precision
acc = accuracy_score(db_val["labels"], preds)
print("[INFO] score: {}".format(acc))

# serialize the model to disk
#print("[INFO] saving model...")
#f = open(args["model"], "wb")
#f.write(pickle.dumps(model.best_estimator_))
#f.close()

# close the database
db.close()
db_val.close()