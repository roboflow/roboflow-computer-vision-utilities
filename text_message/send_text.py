import cv2
import time
import io
import cv2
import requests
from PIL import Image, ImageOps
from requests_toolbelt.multipart.encoder import MultipartEncoder
import math
from twilio.rest import Client#remember to first pip install twilio


# Your Account SID from twilio.com/console
account_sid = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# Your Auth Token from twilio.com/console
auth_token  = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
photos_to_take = 5

# Taking and saving images from a webcam stream
for x in range(photos_to_take):
# Change '0' to '1' or '2' if it cannot find your webcam
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    photo_path = ''.join(['RF_project/webcamphoto',str(x+1),'.jpg'])
    cv2.imwrite(photo_path, frame)
    video.release
    time.sleep(3)


# mirror the images
# Flip the images to match the video format of my labeled images.
for x in range(photos_to_take):
    im = Image.open(''.join(['RF_project/webcamphoto',str(x+1),'.jpg']))
    mirror_image = ImageOps.mirror(im)
    mirror_image.save(''.join(['RF_project/webcamphoto',str(x+1),'.jpg']))

# Load Image with PIL
response = [None] * photos_to_take
for x in range (photos_to_take):
   photo_path = ''.join(['RF_project/webcamphoto',str(x+1),'.jpg'])
   img = cv2.imread(photo_path)
   image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   pilImage = Image.fromarray(image)
 
   # Convert to JPEG Buffer
   buffered = io.BytesIO()
   pilImage.save(buffered, quality=100, format="JPEG")
 
   # Build multipart form and post request
   m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
 
   response[x] = requests.post(
       "https://detect.roboflow.com/YOUR_MODEL/YOUR_MODEL_ID?api_key=YOUR_API_KEY&confidence=.40",
       data=m, headers={'Content-Type': m.content_type}).json()

# See inference results
print(response)

def post_process(response):
    # Post processing - looking at average count of objects in the images, rounding up. 
    response_str = str(response)
    player_count = math.ceil(response_str.count("tennis")/photos_to_take)
    court_count = math.ceil(response_str.count("court")/photos_to_take)


    # Post processing - looking at average count of objects in the images, rounding up. 
    response_str = str(response)
    player_count = math.ceil(response_str.count("tennis")/photos_to_take)
    court_count = math.ceil(response_str.count("court")/photos_to_take)

    # Model used in this example: https://universe.roboflow.com/mixed-sports-area/tennis-court-checker/

def send_text(player_count, court_count):
    court_phrase = "courts."
    if court_count == 1:
        court_phrase = " court."
    
    player_phrase = " tennis players detected on "
    if player_count == 1:
        player_phrase = " tennis player detected on "
    
    message_body = str(
        "There are " + str(player_count) + player_phrase + str(court_count) + court_phrase)
    
    print(message_body)

    client = Client(account_sid, auth_token)
    
    message = client.messages.create(to="+XXXXXXXXXX", from_="+XXXXXXXXXX",body=message_body)
    
    print(message.sid, court_phrase, player_phrase)
