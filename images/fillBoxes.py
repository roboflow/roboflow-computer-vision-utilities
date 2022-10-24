import os, glob, shutil
from roboflow import Roboflow
import json
import cv2


def fillBoxes(model, img_path, printJson = True, save_img = True, confidence = 40, overlap = 30):
    """
    This function fills the bounding boxes on images passed for prediction to your Roboflow model endpoint and also saves the "unfilled" prediction image\n
    :param model: Roboflow model object for the first inference pass\n
    :param img_path: string, path to the image for inference\n
    :param printJson: bool, default: True - prints the JSON values of the predictions from the second inference pass\n
    :param save_img: bool, default: True - saves the prediction image [crop] from the second inference pass\n
    :param confidence: int, default: 40 - minimum confidence level (%) to return predictions\n
    :param overlap: int, default: 30 - maximum prediction overlap (% bounding boxes can overlap prior to being considered the same box/detection)
    """
    ## for a directory (folder) of images
    if os.path.isdir(img_path):
        raw_data_location = img_path
        for raw_data_ext in ['.jpg', '.jpeg', 'png']:
            globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
            for img_file in globbed_files:
                img = cv2.imread(img_file)
                ## perform inference on the selected image
                predictions = model.predict(img_file, confidence=confidence,
                    overlap=overlap)
                #predictions_json = predictions.json()
                original_file = os.path.basename(img_file).split('/')[-1]
                predictions.save(os.curdir + f"/inference_images/inferred/{original_file}")
                ## drawing bounding boxes with the Pillow library
                ## https://docs.roboflow.com/inference/hosted-api#response-object-format
                for bounding_box in predictions:
                    # defining bounding box area and saving class name and confidence scores
                    x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
                    x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
                    y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
                    y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
                    # class_name = bounding_box['class']
                    # confidence_score = bounding_box['confidence']#confidence score of prediction
                    start_point = (int(x0), int(y0))# bounding box start point (pt1)
                    end_point = (int(x1), int(y1))# bounding box end point (pt2)

                    ## draw/place filled bounding boxes on image
                    cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)

                    if save_img:
                        if os.path.exists(os.curdir + f"/inference_images/fillBoxes"
                            ) is False:
                            os.mkdir(os.curdir + f"/inference_images/fillBoxes")
                        
                        filename = original_file
                        save_loc = f'./inference_images/fillBoxes/' + filename
                        cv2.imwrite(save_loc, img)

                        print(f'Success! Saved to {save_loc}')

                    if printJson:
                        print(f'\n{bounding_box}')
                        
    ## runs if there is only 1 image file in the ./inference_images directory
    elif os.path.isfile(img_path):
        img = cv2.imread(img_path)
        ## perform inference on the selected image
        predictions = model.predict(img_path, confidence=confidence,
            overlap=overlap)
        #predictions_json = predictions.json()
        original_file = os.path.basename(img_path).split('/')[-1]
        predictions.save(os.curdir + f"/inference_images/inferred/{original_file}")
        ## drawing bounding boxes with the Pillow library
        ## https://docs.roboflow.com/inference/hosted-api#response-object-format
        for bounding_box in predictions:
            ## defining bounding box area and saving class name and confidence scores
            x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
            x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
            y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
            y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
            start_point = (int(x0), int(y0))# bounding box start point (pt1)
            end_point = (int(x1), int(y1))# bounding box end point (pt2)

            ## draw/place bounding boxes on image
            cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)
        
            if save_img:
                if os.path.exists(os.curdir + f"/inference_images/fillBoxes"
                    ) is False:
                    os.mkdir(os.curdir + f"/inference_images/fillBoxes")
                
                filename = original_file
                save_loc = f'./inference_images/fillBoxes/' + filename
                cv2.imwrite(save_loc, img)

                print(f'Success! Saved to {save_loc}')

            if printJson:
                print(f'\n{bounding_box}')
                
    else:
        return print('Please input a valid path to an image or directory (folder)')


## load config file for the models
with open(os.pardir + '/roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    f.close()

## obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
## create Roboflow object: https://docs.roboflow.com/python
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
project = workspace.project(ROBOFLOW_MODEL_ID)
version = project.version(ROBOFLOW_VERSION_NUMBER)
model = version.model

## creating a directory to add images we wish to infer
if os.path.exists(os.curdir + '/inference_images') is False:
    os.mkdir(os.curdir + '/inference_images')

## creating directory to place Roboflow prediction images
if os.path.exists(os.curdir + '/inference_images/inferred') is False:
    os.mkdir(os.curdir + '/inference_images/inferred')

## creating directory to place prediction images with filled bounding boxes
if os.path.exists(os.curdir + '/inference_images/fillBoxes') is False:
    os.mkdir(os.curdir + '/inference_images/fillBoxes')

for raw_data_ext in ['.jpg', '.jpeg', 'png']:
    globbed_files = glob.glob(os.curdir + '/*' + raw_data_ext)
    for img_file in globbed_files:
        shutil.move(img_file, os.curdir + '/inference_images')

fillBoxes(model, './inference_images', confidence = 40, overlap = 30)
