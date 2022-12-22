from roboflow import Roboflow
import os, sys, shutil
import json
import re
import glob


def count_objects(predictions, target_classes):
    """
    Helper method to count the number of objects in an image for a given class
    :param predictions: predictions returned from calling the predict method
    :param target_class: str, target class for object count
    :return: dictionary with target class and total count of occurrences in image
    """
    object_counts = {x:0 for x in target_classes}
    for prediction in predictions:
        if prediction['class'] in target_classes:
            object_counts[prediction['class']] += 1
        elif prediction['class'] not in target_classes:
            object_counts[prediction['class']] = 1

    present_objects = object_counts.copy()

    for i in object_counts:
        if object_counts[i] < 1:
            present_objects.pop(i)

    return present_objects


## load config file for the models
with open(os.curdir + '/roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    f.close()

# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
project = workspace.project(ROBOFLOW_MODEL_ID)
version = project.version(ROBOFLOW_VERSION_NUMBER)
model = version.model

## creating a directory to add images we wish to infer
if os.path.exists(os.curdir + '/images_to_infer') is False:
    os.mkdir(os.curdir + '/images_to_infer')

for data_ext in ['.jpg', '.jpeg', '.png']:
    globbed_files = glob.glob(os.curdir + '/*' + data_ext)
    for img_file in globbed_files:
        shutil.move(img_file, os.curdir + '/images_to_infer')

file_location = f"{os.curdir + '/images_to_infer'}"
file_extension = ".jpg" # e.g jpg, jpeg, png

globbed_files = glob.glob(file_location + '/*' + file_extension)
## Uncomment the following line to print all class labels in the project
# print(project.classes)

for img_file in globbed_files:
    # perform inference on the selected image
    predictions = model.predict(img_file)
    class_counts = count_objects(predictions, project.classes)
    ## Uncomment the following line to print the individual JSON Predictions
    # print(predictions)
    print('\n', "Class Counts:", '\n')
    print(class_counts)
