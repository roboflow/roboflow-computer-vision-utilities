import cv2
import os, glob
import time
from roboflow import Roboflow
from numpy import mean

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("measure-drill-holes")
model = project.version(5).model

# grab all the .jpg files
extention_images = ".jpg"
get_images = sorted(glob.glob('images/' + '*' + extention_images))

print(get_images)

# font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
org = (25, 25)
fontScale = 2
color = (255, 0, 0)
thickness = 2

box_color = (125, 0, 125)
box_thickness = 3 
box_scale = 4

fpsArray = []
averageFPS = 0

pixel_ratio_array = []
averagePR = []

try:
    for image_paths in get_images:

        print(image_paths)

        response = model.predict(image_paths, confidence=40, overlap=30).json()

        frame = cv2.imread(image_paths)

        t0 = time.time()

        pixel_ratio_array = []
        averagePR = []

        for objects in response['predictions']:
            # get prediction_name and confidence of each object
            object_class = str(objects['class'])

            # pull bbox coordinate points
            x0 = objects['x'] - objects['width'] / 2
            y0 = objects['y'] - objects['height'] / 2
            x1 = objects['x'] + objects['width'] / 2
            y1 = objects['y'] + objects['height'] / 2
            box = (x0, y0, x1, y1)

            box_start_point = (int(x0), int(y0))
            box_end_point = (int(x1), int(y1))

            if object_class == "Reference":

                object_class_text_size = cv2.getTextSize(object_class, font, fontScale, thickness)
                object_confidence = str(round(objects['confidence']*100 , 2)) + "%"
                
                reference_inches = 1

                reference_height = objects['height']
                reference_width = objects['width']

                pixel_to_inches = reference_height / reference_inches
                pixel_ratio_array.append(pixel_to_inches)
                averagePR = mean(pixel_ratio_array)

                object_Inches = reference_height / averagePR

                inches_ORG = (int(x0), int(y0-10))

                frame = cv2.putText(frame, 'Inches: ' + str(object_Inches)[:5], inches_ORG, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)

                # draw ground truth boxes
                frame = cv2.rectangle(frame, box_start_point, box_end_point, box_color, box_thickness)

        ratio_weight = 1.10
        averagePR = averagePR * ratio_weight

        target_size = 0.15625

        target_max = target_size * 1.10
        target_min = target_size * 0.9

        for objects in response['predictions']:
            # get prediction_name and confidence of each object
            object_class = str(objects['class'])

            # pull bbox coordinate points
            x0 = objects['x'] - objects['width'] / 2
            y0 = objects['y'] - objects['height'] / 2
            x1 = objects['x'] + objects['width'] / 2
            y1 = objects['y'] + objects['height'] / 2
            box = (x0, y0, x1, y1)

            box_start_point = (int(x0), int(y0))
            box_end_point = (int(x1), int(y1))

            anomaly_detected = False

            box_color = (0, 0, 255)

            if object_class == "Drill Hole":

                object_class_text_size = cv2.getTextSize(object_class, font, fontScale, thickness)
                object_confidence = str(round(objects['confidence']*100 , 2)) + "%"
                    
                hole_inches = 1

                hole_height = objects['height']
                hole_height_THRESHOLD = hole_height * 1.25

                hole_width = objects['width']
                hole_width_THRESHOLD = hole_width * 1.25

                object_Inches = hole_height / averagePR

                if object_Inches < target_max and object_Inches > target_min:
                    box_color = (0, 200, 0)

                if hole_height > hole_width_THRESHOLD:
                    anomaly_detected = True
                    box_color = (0, 200, 255)

                if hole_width > hole_height_THRESHOLD:
                    anomaly_detected = True
                    box_color = (0, 200, 255)

                inches_ORG = (int(x0), int(y0-10))

                frame = cv2.putText(frame, 'Inches: ' + str(object_Inches)[:5], inches_ORG, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)

                # draw ground truth boxes
                frame = cv2.rectangle(frame, box_start_point, box_end_point, box_color, box_thickness)

        # timing: for benchmarking purposes
        t = time.time()-t0

        fpsArray.append(1/t)
        averageFPS = mean(fpsArray)
        averagePR = mean(pixel_ratio_array)

        print("IMAGE CONFIRMED")
        print("PIXEL RATIO: " + str(averagePR) + "\n")

        cv2.imwrite(image_paths[:-3]+"prediction.jpg", frame)
except:
    print("IMAGE ERROR")
    pass