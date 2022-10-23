# roboflow-computer-vision-utilities
Interface with the Roboflow API and Python package for running inference (receiving predictions) from your Roboflow Train computer vision models.

![Roboflow Logo](https://camo.githubusercontent.com/b9468c9d506b644007e50189fd2aa5d5f158b992bb21569222fe3967e608c467/68747470733a2f2f692e696d6775722e636f6d2f6c58436f5674352e706e67)

![Contact Us!](https://i.imgur.com/rBmXoQ1.png)
## [Website](https://docs.roboflow.com/python) • [Docs](https://docs.roboflow.com) • [Blog](https://blog.roboflow.com) • [Twitter](https://twitter.com/roboflow) • [Linkedin](https://www.linkedin.com/company/roboflow-ai) • [Roboflow Universe](https://universe.roboflow.com) • [Knowledge Base](https://help.roboflow.com)

# What is Roboflow?
## **Roboflow** makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. This repo utilizes the official [Roboflow python package](https://docs.roboflow.com/python) that interfaces with the [Roboflow API](https://docs.roboflow.com/inference/hosted-api). Key features of Roboflow:

- Import and Export image datasets into any supported [formats](https://roboflow.com/formats)
- [Preprocess](https://docs.roboflow.com/image-transformations/image-preprocessing)
  and [augment](https://docs.roboflow.com/image-transformations/image-augmentation) data using Roboflow's dataset
  management tools
- Train computer vision models using [Roboflow Train](https://docs.roboflow.com/train) and deploy
  to [production](https://docs.roboflow.com/inference)
- Use [community curated projects](https://universe.roboflow.com/) to start building your own vision-powered products

## [Features by workspace type](https://roboflow.com/pricing)

# Personal and Research Projects - Applying for Additional Account Features:
* https://roboflow.com/community

# Business Projects and POC's - Requesting Additional Account Features:
* https://roboflow.com/sales

# GitHub Repo Structure:
* The `Images` directory contains necessary code for running inference (model predictions) on individual images, and folders (directories) of images
* `predictionUtils.py` is written in an Object-Oriented Programming framework, and contains necessary code for interacting with `Images.py` for saving [customized] images from inference results.
* `twoPass.py`: Code for running inference (model predictions) in "two stages" on images.
  * Ex. Stage 1: object detection (find faces) --> crop the detected areas and send to --> Stage 2: classification (is this a real image or illustrated?)
  * Available in `roboflow-computer-vision-utilities/images`
* `webcam_od.py`: Code for running inference (model predictions) with Object Detection models on webcam feeds
  * Available in `roboflow-computer-vision-utilities/webcam`
* `webcam_classification.py`: Code for running inference (model predictions) with Classification models on webcam feeds
  * Available in `roboflow-computer-vision-utilities/webcam`

[In Progress]:
* `Video.py`: Code for running inference (model predictions) on local video files

## Installation (Dependencies):

To install the Python package, please use `Python 3.6` or higher. We provide three different ways to install the Roboflow
package to use within your own projects.

Install from Source:

```
git clone https://github.com/roboflow-ai/roboflow-computer-vision-utilities.git
cd roboflow-computer-vision-utilities
python3 -m venv env
source env/bin/activate 
pip3 install -r requirements.txt
```

## [Obtaining Your API Key](https://docs.roboflow.com/rest-api#obtaining-your-api-key) | [Locating Project Information](https://docs.roboflow.com/python#finding-your-project-information-manually)

Colab Tutorials Here:
<a href="https://colab.research.google.com/drive/1UxQTtSqxUF2EM-iS0j7hQPFdqC0eW66A?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- Select "File" in the Google Colab menu, and "Save a Copy in Drive" prior to running the notebook

![Saving Your Own Copy of the Colab Notebook](/figures/ColabNotebook_SaveFile.png)

![The Roboflow API JSON Response Object Format](/figures/ResponseObjectFormat_JSON.png)

![Visualized Roboflow API JSON Response Object Format](/figures/Visualized_ResponseObjectFormat_JSON.png)

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

This app was created using [Roboflow](https://roboflow.com) and [Streamlit](https://streamlit.io/).

## Example Code Snippets
The following code snippets are configured to work after executing:
* Single image file:
```
img_path = 'INSERT_PATH_TO_IMG' # .jpg, .jpeg, .png
img = cv2.imread(img_path)
# perform inference on the selected image
predictions = model.predict(img_path, confidence=40,
    overlap=30)
```
* Folder/directory with image files:
```
raw_data_location = "INSERT_PATH_TO_DIRECTORY"
for raw_data_extension in ['.jpg', '.jpeg', 'png']:
## using the following line for raw_data_externsion results in inference on
## specified file types only
# raw_data_extension = ".jpg" # e.g jpg, jpeg, png
    globbed_files = glob.glob(raw_data_location + '/*' + raw_data_extension)
    for img_path in globbed_files:
        img = cv2.imread(img_path)
        predictions = model.predict(img_path, confidence=40, overlap=30)
```

### Drawing Bounding Boxes
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

### Drawing "Filled" Bounding Boxes
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

### Blurring the Contents of Bounding Boxes
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

### Cropping the Contents of Bounding Boxes
```
# set add_label to 'True' if you want the class label on the image
add_label = 'False'
i = 0

for bounding_box in predictions:
  # defining crop area [height_of_cropArea:width_of_cropArea]
  # croppedArea = img[start_row:end_row, start_col:end_col]
  x0 = bounding_box['x'] - bounding_box['width'] / 2#start_column
  x1 = bounding_box['x'] + bounding_box['width'] / 2#end_column
  y0 = bounding_box['y'] - bounding_box['height'] / 2#start row
  y1 = bounding_box['y'] + bounding_box['height'] / 2#end_row
  class_name = bounding_box['class']
  croppedArea = img[int(y0):int(y1), int(x0):int(x1)]

  # if add_label is 'True' add class name with filled background
  if add_label == 'True':
    # position coordinates: start = (x0, y0), end = (x1, y1)
    # color = RGB-value for bounding box color, (0,0,0) is "black"
    # thickness = stroke width/thickness of the box, -1 = fill box
    cv2.rectangle(croppedArea,
      (int(10), int(10)), (int(80), int(40)), color=(0,0,0),
      thickness=-1)
    # write class name on image, and print class name
    cv2.putText(
        croppedArea, # PIL.Image object to place text on
        class_name,#text to place on image
        (20, 20),#location of text in pixels
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, #text font
        fontScale = 0.4,#font scale
        color = (255, 255, 255),#text color in RGB
        thickness=2#thickness/"weight" of text
        )
  # SAVE CROPPED IMAGES
  cv2.imwrite(f'crop{i}_' + os.path.basename(img_path), croppedArea)
  i+=1
```

### Writing and Placing Text
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

### Run model inference on a single image file:
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
### Run model inference on a folder (directory) of image files:
```
raw_data_location = "INSERT_PATH_TO_IMG_DIRECTORY"
for raw_data_extension in ['.jpg', '.jpeg', 'png']:
## using the following line for raw_data_externsion results in inference on
## specified file types only
# raw_data_extension = ".jpg" # e.g jpg, jpeg, png
    globbed_files = glob.glob(raw_data_location + '/*' + raw_data_extension)
    for img_path in globbed_files:
        predictions = model.predict(img_path, confidence=40, overlap=30)
        # save prediction image
        predictions.save(f'inferenceResult_{os.path.basename(img_path)}')
        predictions_json = predictions.json()
        print(predictions_json)
```
