# roboflow-computer-vision-utilities
Interface with the Roboflow API and Python package for running inference (receiving predictions) from your Roboflow Train computer vision models.

![Roboflow Logo](/figures/roboflow-cv-utilities-header.png)

![Contact Us!](https://i.imgur.com/rBmXoQ1.png)
#### [Website](https://docs.roboflow.com/python) • [Docs](https://docs.roboflow.com) • [Blog](https://blog.roboflow.com) • [Twitter](https://twitter.com/roboflow) • [Linkedin](https://www.linkedin.com/company/roboflow-ai) • [Roboflow Universe](https://universe.roboflow.com) • [Knowledge Base](https://help.roboflow.com)

## What is Roboflow?
**Roboflow** makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. This repo utilizes the official [Roboflow python package](https://docs.roboflow.com/python) that interfaces with the [Roboflow API](https://docs.roboflow.com/inference/hosted-api). Key features of Roboflow:

- Import and Export image datasets into any supported [formats](https://roboflow.com/formats)
- [Preprocess](https://docs.roboflow.com/image-transformations/image-preprocessing)
  and [augment](https://docs.roboflow.com/image-transformations/image-augmentation) data using Roboflow's dataset
  management tools
- Train computer vision models using [Roboflow Train](https://docs.roboflow.com/train) and deploy
  to [production](https://docs.roboflow.com/inference)
- Use [community curated projects](https://universe.roboflow.com/) to start building your own vision-powered products

[Features & Workspace Plans](https://roboflow.com/pricing)

Personal and Research Projects - Applying for Additional Account Features:
* https://roboflow.com/community

Business Projects and POC's - Requesting Additional Account Features:
* https://roboflow.com/sales

#### Installation (Dependencies):
Python Version: `3.10>=Python>=3.7`.
Install from Source:
```
git clone https://github.com/roboflow-ai/roboflow-computer-vision-utilities.git
cd roboflow-computer-vision-utilities
python3 -m venv env
source env/bin/activate 
pip3 install -r requirements.txt
```

#### [Obtaining Your API Key](https://docs.roboflow.com/rest-api#obtaining-your-api-key) | [Workspace ID](https://docs.roboflow.com/roboflow-workspaces#workspace-id) | [Project ID](https://docs.roboflow.com/roboflow-workspaces/projects#project-id) | [Model Version Number](https://docs.roboflow.com/roboflow-workspaces/versions#version-number)

## 🤖 📹 Inference Utilities:
| **functionality** | **images** | **video** | **stream** |
|:------------:|:-------------------------------------------------:|:---------------------------:|:---------------------------:|
| Draw Boxes | [![GitHub](https://badges.aleen42.com/src/github.svg)](/images/draw_img.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/video/draw_img.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/video/draw_vid.py) |
| Write Text | [![GitHub](https://badges.aleen42.com/src/github.svg)](/images/writetext_img.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/video/writetext_vid.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/stream/writetext_stream.py) |
| Fill Boxes | [![GitHub](https://badges.aleen42.com/src/github.svg)](/images/fill_img.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/video/fill_vid.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/stream/fill_stream.py) |
| Crop Boxes | [![GitHub](https://badges.aleen42.com/src/github.svg)](/images/crop_img.py) | | |
| Blur Boxes | [![GitHub](https://badges.aleen42.com/src/github.svg)](/images/blur_img.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/video/blur_video.py) | [![GitHub](https://badges.aleen42.com/src/github.svg)](/stream/blur_stream.py) |
| Object Counting | [![GitHub](https://badges.aleen42.com/src/github.svg)](/object_counting.py) <a href="https://blog.roboflow.com/no-code-computer-vision-zapier/" rel=""><img src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649" width=15% alt="Roboflow Blog" /></a> | | |
| Measure Object | [![GitHub](https://badges.aleen42.com/src/github.svg)](/measureObject.py) <a href="https://blog.roboflow.com/no-code-computer-vision-zapier/" rel=""><img src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649" width=15% alt="Roboflow Blog" /></a> | | |
| Send Email | [![GitHub](https://badges.aleen42.com/src/github.svg)](/trigger_power_automate.py) <a href="https://blog.roboflow.com/no-code-computer-vision-zapier/" rel=""><img src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649" width=15% alt="Roboflow Blog" /></a> | | |

#### Video Inference (Classification)
* Model predictions for Classification models running via hosted or local deployment supported by Roboflow Deploy.

[![GitHub](https://badges.aleen42.com/src/github.svg)](/video_classification.py)

#### Webcam Inference (Classification)
* Model predictions for Classification models running via hosted or local deployment supported by Roboflow Deploy.

[![GitHub](https://badges.aleen42.com/src/github.svg)](/webcam_classification.py)

#### Sample Video Frames
* Extract and save video frames by a specified frames per second of video.

[![GitHub](https://badges.aleen42.com/src/github.svg)](/save_vidframes.py)

## 🔁 📈 Active Learning Utilities:
Automate improvement of your dataset by using computer vision and conditional upload logic to determine which images should be directly uploaded to your Roboflow workspace.
* [Active Learning](https://roboflow.com/python/active-learning) (Roboflow Documentation)
* [Active Learning with the Roboflow Python SDK](https://blog.roboflow.com/pip-install-roboflow)
* [Strategies for Active Learning Implementation](https://blog.roboflow.com/computer-vision-active-learning-tips/)
* [Active Learning](https://help.roboflow.com/implementing-active-learning) (Knowledge Base guide)

| **model type** | **images** | **video** |
|:------------:|:-------------------------------------------------:|:---------------------------:|
| Object Detection | [![GitHub](https://badges.aleen42.com/src/github.svg)](---) | [![GitHub](https://badges.aleen42.com/src/github.svg)](---) |

Conditionals - Source Code:

* `roboflow-python/roboflow/util/active_learning_utils.py`: [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/roboflow/roboflow-python/blob/main/roboflow/util/active_learning_utils.py)
* `roboflow-python/roboflow/core/workspace.py` Line 245: [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/roboflow/roboflow-python/blob/02b717f453e95c5cdef8a817914a46f5fd6771a8/roboflow/core/workspace.py#L245)
```
conditionals = {
    "required_objects_count" : 1,
    "required_class_count": 1,
    "target_classes": ["class_name"],
    "minimum_size_requirement" : float('-inf'),
    "maximum_size_requirement" : float('inf'),
    "confidence_interval" : [10,90],
    "similarity_confidence_threshold": .3,
    "similarity_timeout_limit": 3
}
```
* Note: [Filtering out images for upload by similarity](https://blog.roboflow.com/roboflow-inference-server-clip/) is available for paid plans. Please [contact the Roboflow team](https://roboflow.com/sales) for access.

![The Roboflow API JSON Response Object Format](/figures/ResponseObjectFormat_JSON.png)

![Visualized Roboflow API JSON Response Object Format](/figures/Visualized_ResponseObjectFormat_JSON.png)

`twoPass.py`: Code for running inference (model predictions) in "two stages" on images.
```
# default root save location: './inference_images'
# also moves all images in /roboflow-computer-vision-utilities to /roboflow-computer-vision-utilities/inference_images before inferrence
cd images
python3 twoPass.py
```
  * Ex. Stage 1: object detection (find faces) --> crop the detected areas and send to --> Stage 2: classification (is this a real image or illustrated?)
  * To be used after updating the `roboflow_config.json` file in the main directory with your Model Info (Workpsace ID, Model/Project ID, Private API Key and Model Version Number)
  * Available in `roboflow-computer-vision-utilities/images`

`trigger_power_automate.py`: make predictions on images, save the results, and send an email to the email specified in `roboflow_config.json`
```
# default confidence and overlap for predictions: confidence = 40, overlap = 30
cd images
python3 trigger_power_automate.py
```
  * To be used after updating the `roboflow_config.json` file in the main directory with your Model Info (Workpsace ID, Model/Project ID, Private API Key and Model Version Number), and email address to send the inference results to
  * Available in `roboflow-computer-vision-utilities/images`

## Streamlit App for Testing Roboflow Object Detection Models

This app allows you to upload an image to be inferenced by an Object Detection model trained with [Roboflow Train](https://docs.roboflow.com/train)
* [Inference: Hosted API](https://docs.roboflow.com/inference/hosted-api)
* [Response Object Format](https://docs.roboflow.com/inference/hosted-api#response-object-format)
* [Roboflow Python Package](https://docs.roboflow.com/python)

The app example app can be [found here](https://mo-traor3-ai-streamlit-roboflow-model-test-streamlit-app-j8lako.streamlitapp.com/):
* https://mo-traor3-ai-streamlit-roboflow-model-test-streamlit-app-j8lako.streamlitapp.com/
* The code for the app is available in the `streamlit` directory of this repo, in the `streamlit_app.py` file
* The original repo is hosted here: https://github.com/mo-traor3-ai/streamlit-roboflow-model-testing
* [Creating a Basic Streamlit App for Roboflow Inference](https://blog.roboflow.com/how-to-use-roboflow-and-streamlit-to-visualize-object-detection-output/) | [Roboflow Live Video Inference: Streamlit Tutorial](https://www.youtube.com/watch?v=w4fgZg-jb28)

The app will work as-is for inference on individual image files (png, jpeg, jpg formats). If you want to build your own model, you'll need your own API key. [Create a Roboflow account](https://app.roboflow.com) to get your own API key.

The app was created using [Roboflow](https://roboflow.com) and [Streamlit](https://streamlit.io/).

## Example Code Snippets
Run model inference on a single image file:
```
# perform inference on the selected local image file
file_location = "YOUR_IMAGE.jpg"
predictions = model.predict(file_location, confidence=40, overlap=30)

# save prediction image
predictions.save(f'inferenceResult_{os.path.basename(img_file)}')
predictions_json = predictions.json()
print(predictions_json)

# perform inference on the selected HOSTED image file
file_location = "https://www.yourimageurl.com"
prediction_hosted = model.predict(file_location, confidence=40,overlap=30, hosted=True)

# save prediction image
predictions_hosted.save(f'inferenceResult_{file_location.split('www.')[1]}')
predictions_hosted_json = predictions_hosted.json()
print(predictions_hosted_json)
```
Run model inference on a folder (directory) of image files:
```
raw_data_location = "INSERT_PATH_TO_IMG_DIRECTORY"
for raw_data_extension in ['.jpg', '.jpeg', 'png']:
## using the following line for raw_data_externsion results in inference on
## specified file types only
  raw_data_extension = ".jpg" # e.g jpg, jpeg, png
  globbed_files = glob.glob(raw_data_location + '/*' + raw_data_extension)
  for img_path in globbed_files:
      predictions = model.predict(img_path, confidence=40, overlap=30)

      # save prediction image
      predictions.save(f'inferenceResult_{os.path.basename(img_path)}')
      predictions_json = predictions.json()
      print(predictions_json)
```
Single image file:
```
img_path = 'INSERT_PATH_TO_IMG' # .jpg, .jpeg, .png
img = cv2.imread(img_path)

# perform inference on the selected image
predictions = model.predict(img_path, confidence=40,
    overlap=30)
```
Folder/directory with image files:
```
raw_data_location = "INSERT_PATH_TO_DIRECTORY"

for raw_data_extension in ['.jpg', '.jpeg', 'png']:
## using the following line for raw_data_externsion results in inference on
## specified file types only
  raw_data_extension = ".jpg" # e.g jpg, jpeg, png
  globbed_files = glob.glob(raw_data_location + '/*' + raw_data_extension)
  for img_path in globbed_files:
      img = cv2.imread(img_path)
      predictions = model.predict(img_path, confidence=40, overlap=30)
```
Drawing Bounding Boxes
```
# main bounding box coordinates from JSON response object
# https://docs.roboflow.com/inference/hosted-api#response-object-format
x0 = bounding_box['x'] - bounding_box['width'] / 2
x1 = bounding_box['x'] + bounding_box['width'] / 2
y0 = bounding_box['y'] - bounding_box['height'] / 2
y1 = bounding_box['y'] + bounding_box['height'] / 2

# position coordinates: start = (x0, y0), end = (x1, y1)
# color = RGB-value for bounding box color, (0,0,0) is "black"
# thickness = stroke width/thickness of bounding box
# draw and place bounding boxes
start_point = (int(x0), int(y0))
end_point = (int(x1), int(y1))
cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)
```
Drawing "Filled" Bounding Boxes:
```
# main bounding box coordinates from JSON response object
# https://docs.roboflow.com/inference/hosted-api#response-object-format
x0 = bounding_box['x'] - bounding_box['width'] / 2
x1 = bounding_box['x'] + bounding_box['width'] / 2
y0 = bounding_box['y'] - bounding_box['height'] / 2
y1 = bounding_box['y'] + bounding_box['height'] / 2

# position coordinates: start = (x0, y0), end = (x1, y1)
# color = RGB-value for bounding box color, (0,0,0) is "black"
# thickness = stroke width/thickness of bounding box
# draw and place bounding boxes
start_point = (int(x0), int(y0))
end_point = (int(x1), int(y1))

# setting thickness to -1 --> filled bounding box with the specified color
cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)
```
Blurring the Contents of Bounding Boxes:
```
# rip bounding box coordinates from current detection
# note: infer returns center points of box as (x,y) and width, height
# ----- but pillow crop requires the top left and bottom right points to crop
x0 = prediction['x'] - prediction['width'] / 2
x1 = prediction['x'] + prediction['width'] / 2
y0 = prediction['y'] - prediction['height'] / 2
y1 = prediction['y'] + prediction['height'] / 2
box = [(x0, y0), (x1, y1)]
blur_x = int(prediction['x'] - prediction['width'] / 2)
blur_y = int(prediction['y'] - prediction['height'] / 2)
blur_width = int(prediction['width'])
blur_height = int(prediction['height'])

# region of interest (ROI), or area to blur
roi = img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width]

# ADD BLURRED BBOXES
# set blur to (31,31) or (51,51) based on amount of blur desired
blur_image = cv2.GaussianBlur(roi,(51,51),0)
img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width] = blur_image
```
Cropping the Contents of Bounding Boxes:
```

for bounding_box in predictions:
  # defining crop area [height_of_cropArea:width_of_cropArea]
  # croppedArea = img[start_row:end_row, start_col:end_col]
  x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
  x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
  y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
  y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
  class_name = bounding_box['class']
  croppedArea = img[int(y0):int(y1), int(x0):int(x1)]

  # position coordinates: start = (x0, y0), end = (x1, y1)
  # color = RGB-value for bounding box color, (0,0,0) is "black"
  # thickness = stroke width/thickness of the box, -1 = fill box
  cv2.rectangle(croppedArea,
    (int(10), int(10)), (int(80), int(40)), color=(0,0,0),
    thickness=-1)

  # write class name on image, and print class name
  cv2.putText(
      croppedArea, # cv2 image object to place text on
      class_name,# text to place on image
      (20, 20),# location of text in pixels
      fontFace = cv2.FONT_HERSHEY_SIMPLEX,# text font
      fontScale = 0.4,# font scale
      color = (255, 255, 255),#text color in RGB
      thickness=2# thickness/"weight" of text
      )

  # SAVE CROPPED IMAGES
  cv2.imwrite(f'crop{i}_' + os.path.basename(img_path), croppedArea)
  i+=1
```
Writing and Placing Text:
```
# write and place text
cv2.putText(
    img, # PIL.Image object to place text on
    'placeholder text',#text to place on image
    (12, 12),#location of text in pixels
    fontFace = cv2.FONT_HERSHEY_SIMPLEX, #text font
    fontScale = 0.6,#font scale
    color = (255, 255, 255),#text color in RGB
    thickness=2#thickness/"weight" of text
    )
```