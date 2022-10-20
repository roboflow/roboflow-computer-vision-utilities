import json
import cv2
import base64
import requests
import time

# Check to be sure your config file contains details for a Classification Model trained with Roboflow Train
# https://docs.roboflow.com/train | https://docs.roboflow.com/inference-classification/hosted-api
# load config file:
with open('../roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL_ID = "photo-type-classification"
    ROBOFLOW_VERSION_NUMBER = "5"

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
# (if running locally replace https://classify.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "https://classify.roboflow.com/",
    ROBOFLOW_MODEL_ID, "/",
    ROBOFLOW_VERSION_NUMBER,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    ])

# Get webcam interface via opencv-python
# Change '0' to '1' or '2' if it cannot find your webcam
video = cv2.VideoCapture(0)

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
    # Convert Response to JSON
    predictions = resp.json()
    
    # Add predictions (class label and confidence score) to image
    (text_width, text_height), _ = cv2.getTextSize(
        f"{predictions['top']} Confidence: {predictions['confidence']}",
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)
    
    text_location = (5, text_height)
    
    cv2.putText(img, f"{predictions['top']} | Confidence: {predictions['confidence']}",
                text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(255,255,255), thickness=2)   

    return img, predictions


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
