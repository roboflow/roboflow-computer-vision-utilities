from roboflow import Roboflow
import pandas as pd
import os, glob
import yaml
import json
import csv

# load config file
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
    ANNOTATION_FORMAT = config["ANNOTATION_FORMAT"]

    f.close()

# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
# create Roboflow object: https://docs.roboflow.com/python
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
project = rf.workspace(ROBOFLOW_WORKSPACE_ID).project(ROBOFLOW_MODEL_ID)
version = project.version(ROBOFLOW_VERSION_NUMBER)
dataset = version.download(ANNOTATION_FORMAT)
model = version.model

difference_threshold = 1

cwd = os.getcwd()
prediction_dir = cwd + "/Predictions/"

# set model ID and split it
projectID = model.id
projectID_split = projectID.split("/")

# set name and version from model ID
workspace = projectID_split[0] # project.workspace does not return workspace...
projectName = projectID_split[1]
version = projectID_split[2]

# establish folder naming convertion based on download
folderName = project.name + " " + str(version)
folderName = folderName.replace(" ", "-" )

# loop through files/folder of dataset folder
dataset_path = folderName + "/" # path to train, test, valid folder
get_folders = sorted(glob.glob(dataset_path + '*'))

# convert yaml to config.json file for class_map
with open(dataset_path+'data.yaml', 'r') as file:
    configuration = yaml.safe_load(file)

with open(dataset_path+'config.json', 'w') as json_file:
    json.dump(configuration, json_file)
    
with open(dataset_path+'config.json', 'r') as f:
    json_file = json.load(f)

box_color = (0, 200, 0)
box_thickness = 3 
box_scale = 4

# field names
fields = ['image_url', 'local_path', 'label_count_difference','ground_truth_label_count', 'model_label_count', 'pred_confidence'] 

# name of csv file 
filename = "Annotation_Model_Test.csv"

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 

    # writing the fields 
    csvwriter.writerow(fields) 

# create a class map based on the class names found in the yaml file
class_map = json_file['names']
print(class_map)

# loop through all the files in the dataset_path folder and filter out the train, test, valid folder
for folders in get_folders:
    if "." in folders:
        pass
    else:
        
        folder_split = folders.split("\\")

        # print(folder_split[1])

        # print("Current working directory: {0}".format(cwd))

        try:
            os.mkdir(prediction_dir)
        except:
            pass

        try:
            os.mkdir(prediction_dir+folder_split[1])
        except:
            pass

        image_count = 0

        # grab all the .jpg files
        extention_images = ".jpg"
        get_images = sorted(glob.glob(folders + '/images/' + '*' + extention_images))

        # grab all the .txt files
        extention_annotations = ".txt"
        get_annotations = sorted(glob.glob(folders + '/labels/' + '*' + extention_annotations))

        # loop through all the images in the current folder
        for images in get_images:
            
            image_clean = cv2.imread(images)

            dimensions = image_clean.shape

            # get image dimensions
            image_width = dimensions[1]
            image_height = dimensions[0]

            # split the image path to grad the hash
            image_hash_split = images.split(".")
            image_hash = image_hash_split[-2]

            # reconstruct the roboflow URL with image hash to include in CSV log
            roboflow_image_url = 'https://app.roboflow.com/'+ workspace + '/' + projectName + '/' + str(version) + '/images/' + image_hash + "?predictions=true"

            # get the current annotation that matchs current index for the current image we are on (note: this will not work if dataset contains null photos)
            current_annotation = get_annotations[image_count]

            currentLabel = open(current_annotation, "r")
        
            Lines = currentLabel.readlines()

            # print(Lines)

            # loop through all the lines in the annotation file
            for line in Lines:
            
                split = line.split(' ')

                # print(split[0])

                gt_center_x = float(split[1])*image_width
                gt_center_y = float(split[2])*image_height
                gt_width = float(split[3])*image_width
                gt_height = float(split[4])*image_height

                gt_x0 = gt_center_x - (gt_width / 2)
                gt_y0 = gt_center_y - (gt_height / 2)
                gt_x1 = gt_center_x + (gt_width / 2)
                gt_y1 = gt_center_y + (gt_height / 2)

                gt_box = (gt_x0, gt_y0, gt_x1, gt_y1)

                gt_box_start_point = (int(gt_x0), int(gt_y0))
                # print(gt_box_start_point)
                gt_box_end_point = (int(gt_x1), int(gt_y1))
                # print(gt_box_end_point)

                gt_box_color = (0, 0, 200)

                # draw ground truth boxes
                image_drawn = cv2.rectangle(image_clean, gt_box_start_point, gt_box_end_point, gt_box_color, box_thickness)

            # split the image path to grad the hash
            annotation_hash_split = current_annotation.split(".")
            annotation_hash = annotation_hash_split[2]

            # construct arrays to be used for appending class predictions and annotations
            class_annotation_array = []
            class_prediction_array = []
            confidence_prediction_array = []
            
            # construct counters (might not use)
            correct_predictions = 0
            current_total_annotations = 0

            # if the annotation hash and image hash match then preform prediction on image
            if image_hash == annotation_hash:
                
                print("Successful Match")

                minConfidence = 0

                # perform roboflow model prediction
                response = model.predict(images, confidence=40, overlap=30).json()

                # loop through all the response prediction objects and append their class names to class_prediction_array
                for objects in response['predictions']:
                    
                    # get prediction_name and confidence of each object
                    predicted_name = objects['class']
                    confidence = objects['confidence']

                    x0 = objects['x'] - objects['width'] / 2
                    y0 = objects['y'] - objects['height'] / 2
                    x1 = objects['x'] + objects['width'] / 2
                    y1 = objects['y'] + objects['height'] / 2
                    box = (x0, y0, x1, y1)

                    box_start_point = (int(x0), int(y0))
                    # print(box_start_point)
                    box_end_point = (int(x1), int(y1))
                    # print(box_end_point)

                    # append to respective array
                    class_prediction_array.append(predicted_name)
                    confidence_prediction_array.append(confidence)

                    image_drawn = cv2.rectangle(image_drawn, box_start_point, box_end_point, box_color, box_thickness)

                # open corrisponding annotation file and grab all the annotation objects
                with open(current_annotation) as f:
                    for line in f.readlines():

                        # split lines by space to isolate class ID
                        line_split = line.split(" ")

                        # define class_annotation_ID as first index of YOLOv7 annotation
                        class_annotation_ID = line_split[0]

                        # use class_annotation_ID to get corrisponding class name in class map
                        class_name = class_map[int(class_annotation_ID)]
                        
                        # append class name to class_annotation_array
                        class_annotation_array.append(class_name)

            # get length of prediction arrays, calculate differences, and find min confidence
            numberOfPredictions = len(class_prediction_array)
            numberOfAnnotations = len(class_annotation_array)
            numberOfDifferences = abs(numberOfPredictions - numberOfAnnotations)

            try:
                minConfidence = min(confidence_prediction_array)
            except:
                pass
            
            if numberOfDifferences >= difference_threshold:  
                cv2.imwrite(prediction_dir + folder_split[1] + "/" + image_hash + ".jpg", image_drawn)
                local_path = str(prediction_dir + folder_split[1] + "/" + image_hash + ".jpg")
                print()
                print("FAILED - IMAGE WROTE: " + local_path)
            else:
                local_path = "PASSED THRESHOLD"
                print()
                print(local_path)
            
            # isolate unique names for future quality checking
            uniqueClasses = set(class_annotation_array)

            # print number of predictions vs ground truth (log to CSV)
            print()
            print("Roboflow URL: " + roboflow_image_url + " | Number of Differences: " + str(numberOfDifferences) + " | Number of predictions: " + str(numberOfPredictions) + " | Number of Annotations: " + str(numberOfAnnotations) + " | Minumum Confidence: " + str(minConfidence))
            print()

            # name of csv file 
            csv_filename = "Annotation_Model_Test.csv"

            # Dictionary
            dict = {"image_url":str(roboflow_image_url), "local_path":str(local_path), "label_count_difference":str(numberOfDifferences), "ground_truth_label_count":str(numberOfAnnotations),"model_label_count":str(numberOfPredictions), "pred_confidence":str(minConfidence)}

            # writing to csv file 
            with open(csv_filename, 'a') as csvfile: 
                # creating a csv writer object 
                dict_object = csv.DictWriter(csvfile, fieldnames=fields) 

                # write dict to CSV
                dict_object.writerow(dict)

            for uClass in uniqueClasses:
                correct_predictions += 1

            image_count += 1

with open(csv_filename) as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=",")
    sortedlist = sorted(spamreader, key=lambda row:(row['label_count_difference'], row['pred_confidence']), reverse=True)

with open(csv_filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in sortedlist:
        writer.writerow(row)

df = pd.read_csv(csv_filename)
df.to_csv(csv_filename, index=False)