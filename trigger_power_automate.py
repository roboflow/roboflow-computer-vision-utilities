from roboflow import Roboflow
import argparse
from datetime import datetime
import json
import requests
import base64
import cv2
import os, glob
import time


def load_config(email_address=''):
    ## load config file for the models
    with open(os.pardir + '/roboflow_config.json') as f:
        config = json.load(f)

        global ROBOFLOW_API_KEY
        global ROBOFLOW_WORKSPACE_ID
        global ROBOFLOW_MODEL_ID
        global ROBOFLOW_VERSION_NUMBER

        ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
        ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
        ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
        ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
        if config["EMAIL"]:
            EMAIL = config["EMAIL"]
        elif email_address != '':
            EMAIL = email_address
        else:
            print('Please Enter a Valid Email Address to send your prediction results to.')

        f.close()

        return EMAIL


def run_inference(send_address):
    ## obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
    ## create Roboflow object: https://docs.roboflow.com/python
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
    project = workspace.project(ROBOFLOW_MODEL_ID)
    version = project.version(ROBOFLOW_VERSION_NUMBER)
    model = version.model

    # email to send the results to
    email = send_address

    # grab all the .jpg files
    extention_images = ".jpg"
    get_images = sorted(glob.glob('*' + extention_images))

    # box color and thickness
    box_color = (125, 0, 125)
    box_thickness = 3 
    box_scale = 4

    # font settings
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    org = (25, 25)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    try:
        for image_paths in get_images:

            object_count = 0

            now = datetime.now() # current date and time
            date_time = now.strftime("%m-%d-%Y %H-%M-%S") # generate timestamp

            frame = cv2.imread(image_paths)

            response = model.predict(image_paths, confidence=40, overlap=30).json()

            t0 = time.time()

            for objects in response['predictions']:
                # get prediction_name and confidence of each object
                object_class = str(objects['class'])
                object_confidence = str(round(objects['confidence']*100 , 2)) + "%"

                # pull bbox coordinate points
                x0 = objects['x'] - objects['width'] / 2
                y0 = objects['y'] - objects['height'] / 2
                x1 = objects['x'] + objects['width'] / 2
                y1 = objects['y'] + objects['height'] / 2
                box = (x0, y0, x1, y1)

                box_start_point = (int(x0), int(y0))
                box_end_point = (int(x1), int(y1))

                object_count += 1

                inches_ORG = (int(x0), int(y0-10))

                frame = cv2.putText(frame, 'Class: ' + str(object_class), inches_ORG, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)

                # draw ground truth boxes
                frame = cv2.rectangle(frame, box_start_point, box_end_point, box_color, box_thickness)

            # timing: for benchmarking purposes
            t = time.time()-t0

            cv2.imwrite(image_paths[:-3]+"prediction.jpg", frame)

            print("IMAGE CONFIRMED")

            with open(image_paths[:-3]+"prediction.jpg", "rb") as image_prediction_file:
                encoded_string_prediction = base64.b64encode(image_prediction_file.read())
                encoded_string_prediction = encoded_string_prediction.decode('utf-8')
                # print(encoded_string_prediction)

            with open(image_paths, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                encoded_string = encoded_string.decode('utf-8')
                # print(encoded_string)

            url = "https://prod-66.westus.logic.azure.com:443/workflows/42007a30a5954e2ab0af95ac083d58f3/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=F-3LJPi8ocpH49SM_9sI4ESU-KwDXsYFauvpJztQYXI"
            myobj = {'email': str(email), 'last_class': str(object_class), 'most_class': str(object_class), 'average_confidence': str(object_confidence), 'number_of_objects': str(object_count), 'timestamp': str(date_time), 'source_base_64': str(encoded_string), 'tested_base_64': str(encoded_string_prediction)}

            x = requests.post(url, json = myobj)

            print(x.text)

    except:
        print("IMAGE ERROR")
        pass


confirm_send_address = load_config()
infer = run_inference(confirm_send_address)
