# Workspace variables
import os

os.chdir("\\".join(os.path.abspath(__file__).split('\\')[:-1])) # needed for loading .py files

tenant_id = os.environ.get("TENANT_ID", "72f988bf-86f1-41af-91ab-2d7cd011db47")
subscription_id = os.environ.get("SUBSCRIPTION_ID", "b7b2c9e5-8168-4d67-af4f-92602f66ac49")
resource_group = os.environ.get("RESOURCE_GROUP", "mmAmlsWkgrp02")
workspace_region = os.environ.get("WORKSPACE_REGION", "eastus2")
workspace_name = os.environ.get("WORKSPACE_NAME", "mmAmlsWksp02")


# Training preparation (not training execution) variables
remote_data_folder = os.environ.get("REMOTE_DATA_FOLDER", "data_mnist")

# local training data that will be downloaded by the training script from the storage account
local_data_to_download_folder = os.environ.get("LOCAL_DATA_TO_DOWNLOAD_FOLDER", "data_to_download")


# Workspace definition
from azureml.core.authentication import InteractiveLoginAuthentication
authorization = InteractiveLoginAuthentication(tenant_id = tenant_id)

from azureml.core import Workspace
ws = Workspace.create(
    auth=authorization,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    name = workspace_name,              
    create_resource_group = True,
    exist_ok = True)
ws.write_config()

# Compute target variables
compute_target_name = os.environ.get("COMPUTE_TARGET_NAME", "mm-train-cpu-1") 
vm_size = os.environ.get("VM_SIZE", "STANDARD_DS3_V2") # STANDARD_NC6 is GPU-enabled


# Compute target definition
from azureml.core.compute import ComputeTarget, AmlCompute
import time, numpy as np
start = time. time()
if compute_target_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_target_name]
    if type(compute_target) is AmlCompute:
        print('Found compute target: ' + compute_target_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size=vm_size,
        min_nodes=0,
        max_nodes=4)
    
    compute_target = ComputeTarget.create( # create the compute target
        ws, compute_target_name, provisioning_config)
    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=20)
    
# For a more detailed view of current cluster status, use the 'status' property
print(compute_target.status.serialize())
end = time. time()
duration = np.int(end - start)
print('Remote Training Cluster Creation', duration)



# ***************** 1. CREATE PythonScriptStep1: DATA PREPARATION **************************

# ...which needs a RunConfiguration object that specifies the right Conda/Pip dependencies. So we proceed as follows in 3 steps:

# 1.A) Create PipelineData Object datapreparation_output
from azureml.pipeline.core import PipelineData
is_directory=False # it's a file where we save the prepared dataframe
default_datastore = ws.get_default_datastore()
datapreparation_output = PipelineData('datapreparation_output', datastore=default_datastore, is_directory=is_directory)

# 1.B) Create the dependency object with mlextend package https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.conda_dependencies.condadependencies?view=azure-ml-py:
from azureml.core.environment import CondaDependencies
conda_dep_prep = CondaDependencies()
conda_dep_prep.add_pip_package("mlxtend==0.17.2") # or conda_dep.add_conda_package("mlxtend==0.17.2")

# 1.C) Create the RunConfiguration object:
from azureml.core import RunConfiguration
run_config_prep = RunConfiguration(conda_dependencies=conda_dep_prep)

# 1.D) Create the PythonScriptStep
from azureml.pipeline.steps import PythonScriptStep
data_preparation_step = PythonScriptStep(
    name = "1: Data preparation",
    script_name = "1-data_preparation.py",
    compute_target = compute_target,
    runconfig=run_config_prep,
    arguments = [
        "--remoteDataFolder", remote_data_folder,
        "--localDataFolder", local_data_to_download_folder,
        "--datapreparation_output", datapreparation_output,
        "--is_directory", 'aaa' if is_directory else ''], #  All non-empty strings have a True boolean value
    outputs = [datapreparation_output],
    source_directory = './local_scripts/')




#
# ***************** CREATE PythonScriptStep2: TRAINING **************************
#

# 2.A) Create PipelineData Object datatrain_output
from azureml.pipeline.core import PipelineData
is_directory=False # it's a file where we save the prepared dataframe
default_datastore = ws.get_default_datastore()
datatrain_output = PipelineData('datatrain_output', datastore=default_datastore, is_directory=is_directory)

# 2.B) Create the Estimator object 
from azureml.train.estimator import Estimator

pip_packages=['numpy==1.18.2', 'sklearn==0.0', 'mlxtend==0.17.2', 'joblib==0.14.1']

est = Estimator(
    source_directory='./local_scripts2/',
    entry_script='2-train.py',
    pip_packages=pip_packages,
    compute_target=compute_target
    )

# 2.C) Create the EstimatorStep object
from azureml.pipeline.steps import EstimatorStep

train_step = EstimatorStep(
    name="2: Training",
    estimator=est,
    estimator_entry_script_arguments=[
        "--reg", 0.8,
        "--datapreparation_output", datapreparation_output,
        "--datatrain_output", datatrain_output,
        "--is_directory", 'aaa' if is_directory else ''], #  All non-empty strings have a True boolean value],
    inputs=[datapreparation_output],
    outputs=[datatrain_output],
    compute_target=compute_target)


#
# ***************** CREATE PythonScriptStep3: MODEL REGISTRATION **************************
#
# 3.A) Create PipelineData Object modelregistration_output
from azureml.pipeline.core import PipelineData
is_directory=False # it's a file where we save the details of the registered model
default_datastore = ws.get_default_datastore()
modelregistration_output = PipelineData('modelregistration_output3', datastore=default_datastore, is_directory=is_directory)


# 3.C) Create the RunConfiguration object:
from azureml.core import RunConfiguration
run_config_prep = RunConfiguration(conda_dependencies=conda_dep_prep)

# 3.D) Create the PythonScriptStep
from azureml.pipeline.steps import PythonScriptStep
model_registration_step = PythonScriptStep(
    name = "3: Model Registration",
    script_name = "3-model_registration.py",
    compute_target = compute_target,
    #runconfig=run_config_prep,
    arguments = [
        "--model_name", "mauromi_model_name",
        "--datatrain_output", datatrain_output,
        "--modelregistration_output", modelregistration_output,
        "--is_directory", 'aaa' if is_directory else ''], #  All non-empty strings have a True boolean value
    inputs = [datatrain_output],
    outputs= [modelregistration_output],
    source_directory = './local_scripts3/')



# ***************** FINALIZATION **************************
# ***************** FINALIZATION **************************
# ***************** FINALIZATION **************************

#
# ***************** CREATE THE PIPELINE WITHIN THE EXPERIMENT **************************

experiment_name=os.environ.get("EXPERIMENT_NAME", "mauromiExperiment25")

from azureml.pipeline.core import Pipeline
my_pipeline = Pipeline(workspace=ws, steps=[data_preparation_step, train_step, model_registration_step]) 

my_published_pipeline=my_pipeline.publish(name="mauromi Pipeline in MLOPS", description="very interesting")

from azureml.core import Experiment
pipeline_run = Experiment(ws, experiment_name).submit(my_published_pipeline)