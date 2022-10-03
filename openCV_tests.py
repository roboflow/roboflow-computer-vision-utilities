import cv2
import requests
import base64
import io
from PIL import Image
from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("face-detection-mik1i")
model = project.version(7).model

cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(-1, cv2.CAP_V4L)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_color = (255, 255, 255)
font_thickness = 1
font_scale = 1

box_color = (0, 0, 0)
box_thickness = 1
box_scale = 1

ASCII_MODE = False
BASE64_MODE = False
FRAME_MODE = True

while cap.isOpened():
    ret, frame = cap.read()

    if ASCII_MODE == True:
        retval, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer)
        img_str = img_str.decode("ascii")
        print("ASCII MODE: TRUE")
        print(model.predict(img_str, confidence=40, overlap=30).json())
    elif BASE64_MODE == True:
        retval, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer)
        print("BASE64 MODE: TRUE")
        print(model.predict(img_str, confidence=40, overlap=30).json())
    elif FRAME_MODE == True:
        response = model.predict(frame, confidence=40, overlap=30).json()
        # loop through all the response prediction objects and append their class names to class_prediction_array
        for objects in response['predictions']:
            # get prediction_name and confidence of each object
            object_class = str(objects['class'])
            object_class_text_size = cv2.getTextSize(object_class, font, font_scale, font_thickness)
            print("Class: " + object_class)
            object_confidence = str(round(objects['confidence']*100 , 2)) + "%"
            print("Confidence: " + object_confidence)

            # pull bbox coordinate points
            x0 = objects['x'] - objects['width'] / 2
            y0 = objects['y'] - objects['height'] / 2
            x1 = objects['x'] + objects['width'] / 2
            y1 = objects['y'] + objects['height'] / 2
            box = (x0, y0, x1, y1)
            print("Bounding Box Cordinates:" + str(box))

            object_text_confidence = (int(x0)+object_class_text_size[0][0], int(y0)-10)
            class_font_start_point = (int(x0), int(y0)-10)
            box_start_point = (int(x0), int(y0))
            box_end_point = (int(x1), int(y1))

            frame = cv2.rectangle(frame, box_start_point, box_end_point, box_color, box_thickness)
            frame = cv2.putText(frame, object_class, class_font_start_point, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, " - " + object_confidence, object_text_confidence, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    else:
        img_str = 'temp.jpg'
        cv2.imwrite(img_str, frame)
        print("JPG MODE: TRUE")
    
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()