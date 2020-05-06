# stdout will be printed in azureml-logs/80_driver_log.txt and driver_log

from azureml.core import Run
run = Run.get_context() # get hold of the current run 

import argparse, numpy as np, os
# let user feed in 4 parameters: the location of the data files (container+folder from datastore), 
# the regularization rate of the logistic regression algorythm and the model name
parser = argparse.ArgumentParser()

parser.add_argument('--remoteDataFolder', type=str, dest='remote_data_folder', help='remote_data_folder')
parser.add_argument('--localDataFolder', type=str, dest='local_data_folder', help='local_data_folder')
parser.add_argument('--regularizationRate', type=float, dest='reg', default=0.01, help='regularization rate')
parser.add_argument('--modelName', type=str, dest='model_name', default='my_model', help='model name')
args = parser.parse_args()

remote_data_folder = args.remote_data_folder
print  ('Remote data folder:', remote_data_folder)
run.log('Remote data folder', remote_data_folder)

local_data_folder = args.local_data_folder
print  ('Local data folder:', local_data_folder)
run.log('Local data folder', local_data_folder)

reg = args.reg
print  ('Regularization rate:', reg)
run.log('Regularization rate', reg)

model_name = args.model_name
print  ('Model name (template):', model_name)
run.log('Model name (template)', model_name)

#===========================================


# DOWNLOAD TRAINING DATA FROM REMOTE STORAGE TO THE TRAINING CLUSTER

ds=run.experiment.workspace.get_default_datastore()

ds.download(
    prefix=remote_data_folder,    
    target_path=local_data_folder,
    overwrite=True,
    show_progress=True
)

print  ('Remote Data folder:', remote_data_folder)
run.log('Remote Data folder', remote_data_folder)

print  ('Local Data folder :', local_data_folder)
run.log('Local Data folder', local_data_folder)

train_images_path=os.path.join(local_data_folder,remote_data_folder, 'train-images.idx3-ubyte')
print  ('train_images_path :', train_images_path)
run.log('train_images_path', train_images_path)

train_labels_path=os.path.join(local_data_folder,remote_data_folder, 'train-labels.idx1-ubyte')
print  ('train_labels_path :', train_labels_path)
run.log('train_labels_path', train_labels_path)

test_images_path=os.path.join(local_data_folder,remote_data_folder, 't10k-images.idx3-ubyte')
print  ('test_images_path  :', test_images_path)
run.log('test_images_path', test_images_path)

test_labels_path=os.path.join(local_data_folder,remote_data_folder, 't10k-labels.idx1-ubyte')
print  ('test_labels_path  :', test_labels_path)
run.log('test_labels_path', test_labels_path)

#===========================================


# DATA PREPARATION

from mlxtend.data import loadlocal_mnist

# in X_train, each one of the 60K rows (like X_train[0]) contains 28X28=784 integer numbers [0:255]
X_train, y_train = loadlocal_mnist(
    images_path=train_images_path, 
    labels_path  =train_labels_path)


X_test, y_test = loadlocal_mnist(
    images_path=test_images_path, 
    labels_path=test_labels_path)

X_train=X_train/255.0
y_train=y_train.reshape(-1)

X_test=X_test/255.0
y_test=y_test.reshape(-1)


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
run.log('Single Run Accuracy' ,np.float(acc))

#===========================================

# MODEL EXPORT

# This training script saves the model into a local directory named 'outputs' within the trainig cluster. 
# REMEMBER: anything written in the outputs folder of a training cluster VM is automatically uploaded into the workspace on
# ws.get_default_datastore().account_name/File Shares/azureml/<run.id>/outputs, for example:
# mmmlsdkwstorageiuydyjmq/File Shares/azureml/mauromiLogisticExperiment001_1544291191540/outputs

os.makedirs('outputs', exist_ok=True)
import joblib

from datetime import datetime
model_name_final=(datetime.today().strftime('%Y-%m-%d-%H-%M-%S_') + model_name)[:28] # 32 including extension

# os.getcwd() in the next instruction may be omitted, I leave to recall precisely where it is:
model_relative_path=os.path.join('outputs', model_name_final+'.pkl') # path + file

print  ('Batch Run Model name:', model_name_final) # without path
run.log('Batch Run Model name' , model_name_final)

print  ('Batch Run Model RELATIVE storage path:', model_relative_path)  # with relative path
run.log('Batch Run Model RELATIVE storage path' , model_relative_path)

model_full_path=os.path.join(ds.account_name, "Blob Containers", ds.container_name, "azureml", run.id, model_relative_path)
print  ('Batch Run Model FULL storage path:', model_full_path) # with full path
run.log('Batch Run Model FULL storage path' , model_full_path)

from joblib import dump
dump(value=clf, filename=model_relative_path)

run.complete()
