# stdout will be printed in azureml-logs/80_driver_log.txt and driver_log

from azureml.core import Run
run = Run.get_context() # get hold of the current run 

import argparse, numpy as np, os
# let user feed in 4 parameters: the location of the data files (container+folder from datastore), 
# the regularization rate of the logistic regression algorythm and the model name
parser = argparse.ArgumentParser()

parser.add_argument('--remoteDataFolder', type=str, dest='remote_data_folder', help='remote_data_folder')
parser.add_argument('--localDataFolder', type=str, dest='local_data_folder', help='local_data_folder')
parser.add_argument('--datapreparation_output', type=str, help='datapreparation_output')
parser.add_argument('--is_directory', type=bool, help='is_directory')
args = parser.parse_args()

remote_data_folder = args.remote_data_folder
print  ('Remote data folder:', remote_data_folder)
run.log('Remote data folder', remote_data_folder)

local_data_folder = args.local_data_folder
print  ('Local data folder:', local_data_folder)
run.log('Local data folder', local_data_folder)

datapreparation_output = args.datapreparation_output
print  ('datapreparation_output:', datapreparation_output)
run.log('datapreparation_output', datapreparation_output)

is_directory = args.is_directory
print  ('is_directory:', is_directory)
run.log('is_directory', is_directory)



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

# save arrays to disk 
data = {
    'X_train':  X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test
    }

import pickle
with open(datapreparation_output, 'wb') as f:
    pickle.dump(data, f)

# to read the pickle, please use:
#with open(datapreparation_output, 'rb') as f:
#    b = pickle.load(f)