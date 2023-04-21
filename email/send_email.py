import json
import random
from trigger import execute_trigger
from matplotlib.pyplot import stackplot


# load config file, /roboflow_config.json
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

import cv2
import base64
import numpy as np
import requests
import time

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json"
])

# Get webcam interface via opencv-python
# Change '0' to '1' or '2' if it cannot find your webcam
video = cv2.VideoCapture(0)

# Given an array of predictions, check if there are any 
# predictions that seem to be "stacked" on top of each other.
# If any stacks have 3 or more boxes, increment a counter, which
# keeps track of how many frames so far have been detected for having
# a stack of three or more boxes. 
# If 5 consecutive frames are flagged, return True, and reset our counter.
past_frames_where_stacked = 0
def process_preds(preds):
    def check_stacks(pred, stacks):
        for stack in stacks:
            for box in stack:
                if(pred['x'] > (box['x'] - box['width'] / 2) and
                    pred['x'] < (box['x'] + box['width'] / 2)):
                        stack.append(pred)
                        return True
        return False
    
    stacks = []
    
    # iterate over each detected box. If it's found to be part of an 
    # existing stack, add it to that list. If it's not in any stack, add 
    # it as a new, seperate stack.
    
    for pred in preds:
        if not check_stacks(pred, stacks):
            stacks.append([pred])
            
    print("========================")
    print("Detected " + str(len(stacks)) + " stacks from " + str(len(preds)) + " packages.")
    for i,stack in enumerate(stacks):
        print(f'Stack {i+1} has {len(stack)} packages stacked.')
    
    
    def check_if_any_stacks_over(stacks, threshold):
        for stack in stacks:
            if len(stack) > threshold-1:
                return True
        return False
    
    global past_frames_where_stacked
    if check_if_any_stacks_over(stacks, 3):
        past_frames_where_stacked += 1 
    else:
       past_frames_where_stacked = 0 
    
    if past_frames_where_stacked > 5:
        past_frames_where_stacked = 0
        return True, stacks
    else:
        return False, stacks

# Infer via the Roboflow Infer API and return the result
colors = []
def infer(start, current):
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw
    
    try:
        resp = json.loads(resp.read())
    except:
        print("Could not parse response.")
        
    print(resp)

    preds = resp["predictions"]
    stacked, stacks = process_preds(preds)
    
    original_img = img.copy()
    
    global colors
    while (len(colors)) < len(stacks):
        colors.append((random.randrange(255),random.randrange(255),random.randrange(255)))
    
    # Parse result image
    for idx, stack in enumerate(stacks):
        for box in stack:
            x1 = round(box["x"] - box["width"] / 2)
            x2 = round(box["x"] +  box["width"] / 2)
            y1 = round(box["y"] -  box["height"] / 2)
            y2 = round(box["y"] +  box["height"] / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[idx], 5)
    
    if stacked:
        execute_trigger(img, original_img)  
    
    return img

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()
