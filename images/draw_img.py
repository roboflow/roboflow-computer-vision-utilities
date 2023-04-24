import os
import json
import cv2
import numpy as np
from roboflow import Roboflow

import supervision as sv
from supervision.draw.color import Color
from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator


def load_roboflow_model(api_key, workspace_id, project_id, version_number):

    # authenticate to your Roboflow account and load your model
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_id).project(project_id)
    version = project.version(version_number)
    model = version.model
    
    return project, model

def make_prediction(project, model, image_path, confidence, overlap):

    # load the image and make predictions with your model
    img = cv2.imread(image_path)
    predictions = model.predict(image_path, confidence=confidence, overlap=overlap)
    predictions_json = predictions.json()
    roboflow_xyxy = np.empty((0, 4))
    predicted_classes = []
    for bounding_box in predictions:
        x1 = bounding_box['x'] - bounding_box['width'] / 2
        x2 = bounding_box['x'] + bounding_box['width'] / 2
        y1 = bounding_box['y'] - bounding_box['height'] / 2
        y2 = bounding_box['y'] + bounding_box['height'] / 2
        np.vstack((roboflow_xyxy, [x1, y1, x2, y2]))
        predicted_classes.append(bounding_box['class'])
        
        # class_name = bounding_box['class']
        # confidence = bounding_box['confidence']
        sv_xyxy = Detections(roboflow_xyxy).from_roboflow(
            predictions_json,class_list=list((project.classes).keys()))

    return img, predictions_json, sv_xyxy, predicted_classes

def draw_bounding_boxes(image, sv_xyxy, class_ids, add_labels):

    #set add_labels to True to show the label for each object
    image_with_boxes = BoxAnnotator(
        color=ColorPalette.default(), thickness=2).annotate(image, sv_xyxy, labels=class_ids, skip_label=add_labels)

    return image_with_boxes

def save_image(image, original_image_path, output_directory="results"):

    os.makedirs(output_directory, exist_ok=True)
    filename = os.path.basename(original_image_path)
    output_path = os.path.join(output_directory, f"result_{filename}")
    cv2.imwrite(output_path, image)

    return output_path

def main():
    ## Authentication info to load the model. The config file is located at ../roboflow_config.json
    ## Sample project: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/model/25
    ## Workspace ID: "roboflow-universe-projects", Project ID: "construction-site-safety", Version Number: 25
    with open(os.pardir + '/roboflow_config.json') as f:
        config = json.load(f)

        ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
        ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
        ROBOFLOW_PROJECT_ID = config["ROBOFLOW_PROJECT_ID"]
        ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]

        f.close()
    
    api_key = ROBOFLOW_API_KEY
    workspace_id = ROBOFLOW_WORKSPACE_ID
    project_id = ROBOFLOW_PROJECT_ID
    version_number = ROBOFLOW_VERSION_NUMBER
    project, model = load_roboflow_model(api_key, workspace_id, project_id, version_number)

    # Make a prediction on the specified image file
    image_path = "/path/to/image.jpg"
    confidence = 40
    overlap = 30
    image, predictions_json, pred_sv_xyxy, predicted_classes = make_prediction(
        project, model, image_path, confidence, overlap)

    print(predictions_json)

    ## Set add_labels to False to draw class labels on the bounding boxes
    add_labels = True
    for i in range(len(pred_sv_xyxy)):
        image_with_boxes = draw_bounding_boxes(image, pred_sv_xyxy, predicted_classes, add_labels)

    # Save the image with bounding boxes for the detected objects drawn on them
    output_path = save_image(image_with_boxes, image_path)

    print(f"The image has been processed and saved to {output_path}")

if __name__ == "__main__":
    main()
