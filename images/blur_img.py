from roboflow import Roboflow
import cv2
import os


rf = Roboflow(api_key="INSERT_PRIVATE_API_KEY")
project = rf.workspace("INSERT-WORKSPACE-ID").project("INSERT-PROJECT-ID")
# REPLACE VERSION-NUMBER with (trained) model version number
version = project.version(1)
model = version.model

# perform inference on the selected local image file
file_location = "YOUR_IMAGE.jpg"
predictions = model.predict(file_location, confidence=40, overlap=30)
## save prediction image - roboflow python sdk
# predictions.save(f'inferenceResult_{os.path.basename({file_location})}')
predictions_json = predictions.json()
print(predictions_json)

# drawing bounding boxes with the Pillow library
# https://docs.roboflow.com/inference/hosted-api#response-object-format
img = cv2.imread(file_location)
for bounding_box in predictions:
    x0 = bounding_box['x'] - bounding_box['width'] / 2
    x1 = bounding_box['x'] + bounding_box['width'] / 2
    y0 = bounding_box['y'] - bounding_box['height'] / 2
    y1 = bounding_box['y'] + bounding_box['height'] / 2
    class_name = bounding_box['class']
    confidence = bounding_box['confidence']
    box = (x0, x1, y0, y1)
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

    cv2.imwrite(f'inferenceResult_{os.path.basename({file_location})}', img)
