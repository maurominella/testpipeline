from azureml.core import Run
run = Run.get_context() # get hold of the current run 

import argparse, numpy as np, os
# let user feed in 4 parameters: the location of the data files (container+folder from datastore), 
# the regularization rate of the logistic regression algorythm and the model name
parser = argparse.ArgumentParser()
parser.add_argument('--reg', type=float, help='regularization ate')
parser.add_argument('--datapreparation_output', type=str, help='datapreparation_output')
parser.add_argument('--datatrain_output', type=str, help='datatrain_output')
parser.add_argument('--is_directory', type=bool, help='is_directory')
args = parser.parse_args()

reg = args.reg
print  ('Regularization Rate:', reg)
run.log('Regularization Rate', reg)

datapreparation_output = args.datapreparation_output
print  ('datapreparation_output:', datapreparation_output)
run.log('datapreparation_output', datapreparation_output)

datatrain_output = args.datatrain_output
print  ('datatrain_output:', datatrain_output)
run.log('datatrain_output', datatrain_output)

is_directory = args.is_directory
print  ('is_directory:', is_directory)
run.log('is_directory', is_directory)


# import train and test sets using pickle:
import pickle
with open(datapreparation_output, 'rb') as f:
    data = pickle.load(f)

X_train, y_train, X_test, y_test = [value for key, value in data.items()]

#===========================================


# REAL TRAINING

# requires !pip install sklearn numpy

import time, numpy as np
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0/reg, random_state=42, solver='liblinear', multi_class='ovr')

start = time. time()

# train a logistic regression model with regularization rate of 'reg'
clf.fit(X_train, y_train) # takes 96 seconds on Intel i7-2600K CPU @3.40Ghz with SSD

end = time. time()
training_duration = np.int(end - start)
print  ('Single Run Training Duration:', training_duration)
run.log('Single Run Training Duration' , training_duration)


#===========================================

# MODEL ACCURACY CHECK

# Make predictions using the test set and calculate the accuracy
y_hat = clf.predict(X_test) 

import numpy as np
acc= np.average(y_hat == y_test)
print  ("Single Run Accuracy:", np.float(acc))
run.log('Single Run Accuracy' , np.float(acc))

from joblib import dump
dump(value=clf, filename=datatrain_output)
