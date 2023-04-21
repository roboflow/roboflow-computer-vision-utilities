import cv2
from simplegmail import Gmail#remember to pip install simglegmail
import json

gmail = Gmail() # will open a browser window to ask you to log in and authenticate

# load config file, /roboflow_config.json
with open('roboflow_config.json') as f:
    params = json.load(f)['simplegmail_config']

# Defines a custom action to be taken when enough boxes are stacked.
# We'll be sending an email with a photo of the stacked items.
def execute_trigger(overlayed_image, raw_image):
    cv2.imwrite("overlayed_image.jpg", overlayed_image)
    cv2.imwrite("raw_image.jpg", raw_image)
    print("Image successfully saved! Attempting to send email.")
    
    message = gmail.send_message(**params)
    