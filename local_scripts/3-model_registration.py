from azureml.core import Run
run = Run.get_context() # get hold of the current run


import argparse, numpy as np, os
# let user feed in 4 parameters: the location of the data files (container+folder from datastore), 
# the regularization rate of the logistic regression algorythm and the model name
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='model_name')
parser.add_argument('--datatrain_output', type=str, help='datatrain_output')
parser.add_argument('--modelregistration_output', type=str, help='modelregistration_output')
parser.add_argument('--is_directory', type=bool, help='is_directory')
args = parser.parse_args()

model_name = args.model_name
print  ('model_name:', model_name)
run.log('model_name', model_name)

datatrain_output = args.datatrain_output
print  ('datatrain_output:', datatrain_output)
run.log('datatrain_output', datatrain_output)

modelregistration_output = args.modelregistration_output
print  ('modelregistration_output:', modelregistration_output)
run.log('modelregistration_output', modelregistration_output)

is_directory = args.is_directory
print  ('is_directory:', is_directory)
run.log('is_directory', is_directory)


from azureml.core.model import Model

m = Model.register(model_name=model_name, model_path=datatrain_output, workspace=run.experiment.workspace)

m_serialized = m.serialize()
print  ('model_serial23:', m_serialized)
run.log('model_serial23', m_serialized)

import json
with open(modelregistration_output, 'w') as f:
    f.write(json.dumps(m_serialized))

run.complete()