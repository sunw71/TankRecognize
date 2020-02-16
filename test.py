# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
import h5py



db = h5py.File("datasets/m_tank_test/hdf5/test_features.hdf5", "r")

print("[INFO] load model...")
with open('output/resnetLR.pickle', 'rb') as f:
    model = pickle.load(f)


print("[INFO] evaluating...")
preds = model.predict(db["features"])
print(classification_report(db["labels"], 
                            preds,
                            target_names=db["label_names"]))

#print("[INFO] best hyperparameters: {}".format(model.best_params_))

# compute the raw accuracy with extra precision
acc = accuracy_score(db["labels"], preds)
print("[INFO] score: {}".format(acc))


db.close()