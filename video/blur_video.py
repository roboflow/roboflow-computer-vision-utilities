import json
import cv2
import base64
import numpy as np
import requests
import time


# load config
with open('../roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL_ID, "/",
    ROBOFLOW_VERSION_NUMBER,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    "&stroke=5"
])

# Get webcam interface via opencv-python
# Replace with path to video file
video = cv2.VideoCapture("path/to/video.mp4")

# Infer via the Roboflow Infer API and return the result
def infer():
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
    }, stream=True)
    
    predictions = resp.json()
    detections = predictions['predictions']

    # Parse result image
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Add predictions (bounding box, class label and confidence score) to image
    for bounding_box in detections:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        box = [(x0, y0), (x1, y1)]
        blur_x = int(bounding_box['x'] - bounding_box['width'] / 2)
        blur_y = int(bounding_box['y'] - bounding_box['height'] / 2)
        blur_width = int(bounding_box['width'])
        blur_height = int(bounding_box['height'])
        ## region of interest (ROI), or area to blur
        roi = img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width]

        # ADD BLURRED BBOXES
        # set blur to (31,31) or (51,51) based on amount of blur desired
        blur_image = cv2.GaussianBlur(roi,(51,51),0)
        img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width] = blur_image
        ## draw/place bounding boxes on image
        #start_point = (int(x0), int(y0))
        #end_point = (int(x1), int(y1))
        #cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(
            f"{class_name} | {confidence}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)

        cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + text_width, int(y0) - text_height), color=(0,0,0),
            thickness=-1)
        
        text_location = (int(x0), int(y0))
        
        cv2.putText(img, f"{class_name} | {confidence}",
                    text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                    color=(255,255,255), thickness=2) 

    return img, detections


# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image, detections = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")
    print(detections)

# Release resources when finished
video.release()
cv2.destroyAllWindows()
