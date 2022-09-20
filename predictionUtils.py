from roboflow import Roboflow
import os, sys, re, glob
import cv2
import json
from datetime import date
from PIL import Image, ImageDraw, ImageFont
from images import Images
# from video import Video
# from webcam import Webcam

######################## INFERENCE WITH PYTHON PACKAGE ########################
# load config file
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]

    f.close()

# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
# create Roboflow object: https://docs.roboflow.com/python
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
project = rf.workspace(ROBOFLOW_WORKSPACE_ID).project(ROBOFLOW_MODEL_ID)
version = project.version(ROBOFLOW_VERSION_NUMBER)
model = version.model

file_path = './' # replace with path to a folder (directory) or image file

image_infer = Images.LocalImage(model, file_path)

# print JSON response for predictions
image_infer.makePrediction()

## JSON response for predictions and draw bounding boxes on result image
## default keyword argument values for drawBoxes():
## printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True, view_img = False
image_infer.drawBoxes()

## JSON response for predictions and draw FILLED bounding boxes on result image
## default keyword argument values for drawBoxes():
## printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True, view_img = False
image_infer.drawFilledBoxes()

## JSON response for predictions and draw bounding boxes on result image
## default keyword argument values for drawBoxes():
## printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True, view_img = False
image_infer.blurBoxes()
