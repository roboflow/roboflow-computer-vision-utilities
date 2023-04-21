from roboflow import Roboflow
import cv2
import os


rf = Roboflow(api_key="INSERT_PRIVATE_API_KEY")
project = rf.workspace("INSERT-WORKSPACE-ID").project("INSERT-PROJECT/MODEL-ID")
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
    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))

    ##draw/place bounding boxes on image
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

    cv2.imwrite(f'inferenceResultWriteTxt_{os.path.basename({file_location})}', img)
