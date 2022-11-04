import os, glob, shutil
from roboflow import Roboflow
import json
import cv2


def cropBoxes(model, img_path, printJson = True, save_img = True, confidence = 40, overlap = 30):
    """
    This functio fills the bounding boxes on images passed for prediction to your Roboflow model endpoint and also saves the "unfilled" prediction image\n
    :param model1: Roboflow model object for the first inference pass\n
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
                crop_number = 0
                img = cv2.imread(img_file)
                ## perform inference on the selected image
                predictions = model.predict(img_file, confidence=confidence,
                    overlap=overlap)             
                predictions_json = predictions.json()
                if predictions_json['predictions'] == []:
                    print(f"No predictions for {img_path} at confidence: {confidence} and overlap {overlap}")
                else:
                    original_file = os.path.basename(img_path).split('/')[-1]
                    predictions.save(os.curdir + f"/inference_images/inferred/{original_file}")
                    ## drawing bounding boxes with the Pillow library
                    ## https://docs.roboflow.com/inference/hosted-api#response-object-format
                    for bounding_box in predictions:
                        # defining crop area [height_of_cropArea:width_of_cropArea]
                        # croppedArea = img[start_row:end_row, start_col:end_col]
                        x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
                        x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
                        y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
                        y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
                        class_name = bounding_box['class']
                        croppedArea = img[int(y0):int(y1), int(x0):int(x1)]
                        #confidence_score = bounding_box['confidence']#confidence score of prediction

                        if save_img:
                            if os.path.exists(os.curdir + f"/inference_images/cropBoxes/DetectedAs_{class_name}"
                                ) is False:
                                os.mkdir(os.curdir + f"/inference_images/cropBoxes/DetectedAs_{class_name}")
                            
                            filename = f"Crop{crop_number}_{original_file}"
                            save_loc = f'./inference_images/cropBoxes/DetectedAs_{class_name}/' + filename
                            print(filename, save_loc)
                            print(img_file)
                            cv2.imwrite(save_loc, croppedArea)

                            print(f'Success! Saved to {save_loc}')
                            crop_number+=1

                        if printJson:
                            print(f'\n{bounding_box}')
                        
    ## runs if there is only 1 image file in the ./inference_images directory
    elif os.path.isfile(img_path):
        crop_number = 0
        img = cv2.imread(img_path)
        # perform inference on the selected image
        predictions = model.predict(img_path, confidence=confidence,
            overlap=overlap)
        predictions_json = predictions.json()
        if predictions_json['predictions'] == []:
            print(f"No predictions for {img_path} at confidence: {confidence} and overlap {overlap}")
        else:
            original_file = os.path.basename(img_path).split('/')[-1]
            predictions.save(os.curdir + f"/inference_images/inferred/{original_file}")
            # drawing bounding boxes with the Pillow library
            # https://docs.roboflow.com/inference/hosted-api#response-object-format
            for bounding_box in predictions:
                # defining crop area [height_of_cropArea:width_of_cropArea]
                # croppedArea = img[start_row:end_row, start_col:end_col]
                x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
                x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
                y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
                y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
                class_name = bounding_box['class']
                croppedArea = img[int(y0):int(y1), int(x0):int(x1)]
                #confidence_score = bounding_box['confidence']#confidence score of prediction

                if save_img:
                    if os.path.exists(os.curdir + f"/inference_images/cropBoxes/DetectedAs_{class_name}"
                        ) is False:
                        os.mkdir(os.curdir + f"/inference_images/cropBoxes/DetectedAs_{class_name}")
                    
                    filename = f"Crop{crop_number}_{original_file}"
                    save_loc = f'./inference_images/cropBoxes/DetectedAs_{class_name}/' + filename
                    print(filename, save_loc)
                    cv2.imwrite(save_loc, croppedArea)

                    print(f'Success! Saved to {save_loc}')
                    crop_number+=1

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
if os.path.exists(os.curdir + '/inference_images/cropBoxes') is False:
    os.mkdir(os.curdir + '/inference_images/cropBoxes')

for raw_data_ext in ['.jpg', '.jpeg', 'png']:
    globbed_files = glob.glob(os.curdir + '/*' + raw_data_ext)
    for img_file in globbed_files:
        shutil.move(img_file, os.curdir + '/inference_images')

cropBoxes(model, './inference_images', confidence = 40, overlap = 30)
