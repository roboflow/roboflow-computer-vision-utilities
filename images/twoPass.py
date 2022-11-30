import os, glob, shutil
from roboflow import Roboflow
import json
import cv2


def secondModel(model2, cropped_det_path: str, printJson = True, save_img = True):
    """
    :param model2: Roboflow model object for the second inference pass\n
    :param cropped_det_path: string, path to the cropped detection for inference passed from cropBoxes()\n
    :param printJson: bool, default: True - prints the JSON values of the predictions from the second inference pass\n
    :param save_img: bool, default: True - saves the prediction image [crop] from the second inference pass
    """
    # load image as OpenCV object (array) and scale to classification model Resize (ROBOFLOW_SIZE_MODEL2)
    cropped_img = cv2.imread(cropped_det_path)

    # perform inference on the selected image
    predictions_MODEL2 = model2.predict(cropped_det_path)
    predictions_json_MODEL2 = predictions_MODEL2.json()['predictions'][0]
    
    # https://docs.roboflow.com/inference/hosted-api#response-object-format
    top_class = predictions_json_MODEL2['top']
    #top_confidence = predictions_json_MODEL2['confidence']

    if save_img:
        original_file = os.path.basename(cropped_det_path).split('/')[-1]
        if os.path.exists(os.curdir + f"/inferred_secondPassResult/ClassifiedAs_{top_class}"
            ) is False:
            os.mkdir(os.curdir + f"/inferred_secondPassResult/ClassifiedAs_{top_class}")
        
        filename = original_file
        save_loc = f'./inferred_secondPassResult/ClassifiedAs_{top_class}/' + filename
        cv2.imwrite(save_loc, cropped_img)

        print(f'Success! Second Pass (classification) RESIZED IMAGE FOR INFERENCE saved to {save_loc}')

    if printJson:
        print(f'\n{predictions_json_MODEL2}')


def cropBoxes(model1, model2, img_path, printJson = True, save_img = True, confidence = 40, overlap = 30):
    """
    :param model1: Roboflow model object for the first inference pass\n
    :param model2: Roboflow model object for the second inference pass\n
    :param cropped_det_path: string, path to the cropped detection for inference passed from cropBoxes()\n
    :param printJson: bool, default: True - prints the JSON values of the predictions from the second inference pass\n
    :param save_img: bool, default: True - saves the prediction image [crop] from the second inference pass
    """
    # for a directory (folder) of images
    if os.path.isdir(img_path):
        raw_data_location = img_path
        for raw_data_ext in ['.jpg', '.jpeg', '.png']:
            globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
            for img_file in globbed_files:
                crop_number = 0
                img = cv2.imread(img_file)
                # perform inference on the selected image
                predictions = model1.predict(img_file, confidence=confidence,
                    overlap=overlap)
                #predictions_json = predictions.json()
                original_file = os.path.basename(img_file).split('/')[-1]
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
                        if os.path.exists(os.curdir + f"/inferred_cropBoxes/DetectedAs_{class_name}"
                            ) is False:
                            os.mkdir(os.curdir + f"/inferred_cropBoxes/DetectedAs_{class_name}")
                        
                        filename = f"Crop{crop_number}_{original_file}"
                        save_loc = f'./inferred_cropBoxes/DetectedAs_{class_name}/' + filename
                        cv2.imwrite(save_loc, croppedArea)

                        print(f'Success! First Pass (object detection) saved to {save_loc}')
                        crop_number+=1

                    if printJson:
                        print(f'\n{bounding_box}')
                    
                    secondModel(model2, save_loc)
    
    # runs if there is only 1 image file in the ./inference_images directory
    elif os.path.isfile(img_path):
        crop_number = 0
        img = cv2.imread(img_path)
        # perform inference on the selected image
        predictions = model.predict(img_path, confidence=confidence,
            overlap=overlap)
        #predictions_json = predictions.json()
        original_file = os.path.basename(img_file).split('/')[-1]
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
                original_file = os.path.basename(img_path).split('/')[-1]
                if os.path.exists(os.curdir + f"/inferred_cropBoxes/DetectedAs_{class_name}"
                    ) is False:
                    os.mkdir(os.curdir + f"/inferred_cropBoxes/DetectedAs_{class_name}")
                
                filename = original_file
                save_loc = f'./inferred_cropBoxes/DetectedAs_{class_name}/' + filename
                cv2.imwrite(save_loc, croppedArea)

                print(f'Success! First Pass (object detection) saved to {save_loc}')
                crop_number+=1

            if printJson:
                print(f'\n{bounding_box}')
            
            secondModel(model2, save_loc, x0, y0)
    
    else:
        return print('Please input a valid path to an image or directory (folder)')


# load config file for the models
with open(os.pardir + '/roboflow_config_twopass.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_API_KEY_MODEL2 = config["ROBOFLOW_API_KEY_MODEL2"]
    ROBOFLOW_WORKSPACE_ID_MODEL2 = config["ROBOFLOW_WORKSPACE_ID_MODEL2"]
    ROBOFLOW_MODEL_ID_MODEL2 = config["ROBOFLOW_MODEL_ID_MODEL2"]
    ROBOFLOW_VERSION_NUMBER_MODEL2 = config["ROBOFLOW_VERSION_NUMBER_MODEL2"]
    ROBOFLOW_SIZE_MODEL2 = config["ROBOFLOW_SIZE_MODEL2"]

    f.close()

# obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
# create Roboflow object: https://docs.roboflow.com/python
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
project = workspace.project(ROBOFLOW_MODEL_ID)
version = project.version(ROBOFLOW_VERSION_NUMBER)
model = version.model

rf_MODEL2 = Roboflow(api_key=ROBOFLOW_API_KEY_MODEL2)
workspace_MODEL2 = rf_MODEL2.workspace(ROBOFLOW_WORKSPACE_ID_MODEL2)
project_MODEL2 = workspace_MODEL2.project(ROBOFLOW_MODEL_ID_MODEL2)
version_MODEL2 = project_MODEL2.version(ROBOFLOW_VERSION_NUMBER_MODEL2)
model_MODEL2 = version_MODEL2.model

# creating a directory to add images we wish to infer
if os.path.exists(os.curdir + '/inference_images') is False:
    os.mkdir(os.curdir + '/inference_images')

if os.path.exists(os.curdir + '/inference_images/inferred') is False:
    os.mkdir(os.curdir + '/inference_images/inferred')

for raw_data_ext in ['.jpg', '.jpeg', '.png']:
    globbed_files = glob.glob(os.curdir + '/*' + raw_data_ext)
    for img_file in globbed_files:
        shutil.move(img_file, os.curdir + '/inference_images')

# creating directories to save inference results
for directory in ['/inferred_cropBoxes', '/inferred_secondPassResult']:
    if os.path.exists(os.curdir + directory) is False:
        os.mkdir(os.curdir + directory)

cropBoxes(model, model_MODEL2, './inference_images', confidence = 40, overlap = 30)
