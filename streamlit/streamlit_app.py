import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2
from base64 import decodebytes
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roboflow import Roboflow


## store initial session state values
workspace_id, model_id, version_number, private_api_key = ('', '', '', '')
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = '40'
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = '30'
if 'workspace_id' not in st.session_state:
    st.session_state['workspace_id'] = ''
if 'model_id' not in st.session_state:
    st.session_state['model_id'] = ''
if 'version_number' not in st.session_state:
    st.session_state['version_number'] = ''
if 'private_api_key' not in st.session_state:
    st.session_state['private_api_key'] = ''
if 'include_bbox' not in st.session_state:
    st.session_state['include_bbox'] = 'Yes'
if 'include_class' not in st.session_state:
    st.session_state['include_class'] = 'Show Labels'
if 'box_type' not in st.session_state:
    st.session_state['box_type'] = 'regular'

##########
#### Set up main app logic
##########
def drawBoxes(model_object, img_object, uploaded_file, show_bbox, show_class_label,
              show_box_type, font = cv2.FONT_HERSHEY_SIMPLEX):
    
    collected_predictions = pd.DataFrame(columns=['class', 'confidence', 'x0', 'x1', 'y0', 'y1', 'box area'])
    
    if isinstance(uploaded_file, str):
        img = cv2.imread(uploaded_file)
        # perform inference on the selected image
        predictions = model_object.predict(uploaded_file, confidence=int(st.session_state['confidence_threshold']),
                                           overlap=st.session_state['overlap_threshold'])
    else:
        predictions = model_object.predict(uploaded_file, confidence=int(st.session_state['confidence_threshold']),
                                           overlap=st.session_state['overlap_threshold'])
    
    predictions_json = predictions.json()
    # drawing bounding boxes with the Pillow library
    # https://docs.roboflow.com/inference/hosted-api#response-object-format
    for bounding_box in predictions:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence_score = bounding_box['confidence']
        box = (x0, x1, y0, y1)
        collected_predictions = collected_predictions.append({'class':class_name, 'confidence':confidence_score,
                                            'x0':int(x0), 'x1':int(x1), 'y0':int(y0), 'y1':int(y1), 'box area':box},
                                            ignore_index=True)
        # position coordinates: start = (x0, y0), end = (x1, y1)
        # color = RGB-value for bounding box color, (0,0,0) is "black"
        # thickness = stroke width/thickness of bounding box
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        if show_box_type == 'regular':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    font,#font
                    0.4,#font scale
                    (255,255,255),#text color
                    thickness=1#thickness/"weight" of text
                    )

        if show_box_type == 'fill':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    font,#font
                    0.4,#font scale
                    (255,255,255),#text color
                    thickness=1#thickness/"weight" of text
                    )

        if show_box_type == 'blur':
            if show_bbox == 'Yes':
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)
            
            box = [(x0, y0), (x1, y1)]
            blur_x = int(bounding_box['x'] - bounding_box['width'] / 2)
            blur_y = int(bounding_box['y'] - bounding_box['height'] / 2)
            blur_width = int(bounding_box['width'])
            blur_height = int(bounding_box['height'])
            # region of interest (ROI), or area to blur
            roi = img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width]

            # ADD BLURRED BBOXES
            # set blur to (31,31) or (51,51) based on amount of blur desired
            blur_image = cv2.GaussianBlur(roi,(51,51),0)
            img[blur_y:blur_y+blur_height, blur_x:blur_x+blur_width] = blur_image

            if show_class_label == 'Show Labels':
                # add class name with filled background
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + 40, int(y0) - 20), color=(0,0,0),
                        thickness=-1)
                cv2.putText(img,
                    class_name,#text to place on image
                    (int(x0), int(y0) - 5),#location of text
                    font,#font
                    0.4,#font scale
                    (255,255,255),#text color
                    thickness=1#thickness/"weight" of text
                    )

        # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that 
        # the color is converted from BGR to RGB when going from OpenCV image to PIL image
        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)

    return pil_image, collected_predictions, predictions_json


def run_inference():
    rf = Roboflow(api_key=st.session_state['private_api_key'])
    project = rf.workspace(st.session_state['workspace_id']).project(st.session_state['model_id'])
    project_metadata = project.get_version_information()
    # dataset = project.version(st.session_state['version_number']).download("yolov5")
    version = project.version(st.session_state['version_number'])
    model = version.model

    project_type = st.write(f"#### Project Type: {project.type}")

    for version_number in range(len(project_metadata)):
        try:
            if int(project_metadata[version_number]['model']['id'].split('/')[1]) == int(version.version):
                project_endpoint = st.write(f"#### Inference Endpoint: {project_metadata[version_number]['model']['endpoint']}")
                model_id = st.write(f"#### Model ID: {project_metadata[version_number]['model']['id']}")
                version_name  = st.write(f"#### Version Name: {project_metadata[version_number]['name']}")
                input_img_size = st.write(f"Input Image Size for Model Training (pixels, px):")
                width_metric, height_metric = st.columns(2)
                width_metric.metric(label='Pixel Width', value=project_metadata[version_number]['preprocessing']['resize']['width'])
                height_metric.metric(label='Pixel Height', value=project_metadata[version_number]['preprocessing']['resize']['height'])

                if project_metadata[version_number]['model']['fromScratch']:
                    train_checkpoint = 'Checkpoint'
                    st.write(f"#### Version trained from {train_checkpoint}")
                elif project_metadata[version_number]['model']['fromScratch'] is False:
                    train_checkpoint = 'Scratch'
                    train_checkpoint = st.write(f"#### Version trained from {train_checkpoint}")
                else:
                    train_checkpoint = 'Not Yet Trained'
                    train_checkpoint = st.write(f"#### Version is {train_checkpoint}")
        except KeyError:
            continue

    ## Subtitle.
    st.write('### Inferenced/Prediction Image')
    
    ## Pull in default image or user-selected image.
    if uploaded_file is None:
        # Default image.
        default_img_path = "images/test_box.jpg"
        image = Image.open(default_img_path)
        original_image = image
        open_cv_image = cv2.imread(default_img_path)
        original_opencv_image = open_cv_image
        # Display response image.
        pil_image_drawBoxes, df_drawBoxes, json_values = drawBoxes(model, default_img_path, default_img_path,
                                                                   st.session_state['include_bbox'],
                                                                   st.session_state['include_class'],
                                                                   st.session_state['box_type'])
        
    else:
        # User-selected image.
        image = Image.open(uploaded_file)
        original_image = image
        opencv_convert = image.convert('RGB')
        open_cv_image = np.array(opencv_convert)
        # Convert RGB to BGR: OpenCV deals with BGR images rather than RGB
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # Convert PIL image to byte-string so it can be sent for prediction to the Roboflow Python Package
        b = io.BytesIO()
        image.save(b, format='JPEG')
        im_bytes = b.getvalue() 
        # Display response image.
        pil_image_drawBoxes, df_drawBoxes, json_values = drawBoxes(model, open_cv_image, im_bytes,
                                                                   st.session_state['include_bbox'],
                                                                   st.session_state['include_class'],
                                                                   st.session_state['box_type'])
    
    st.image(pil_image_drawBoxes,
            use_column_width=True)
    # Display original image.
    st.write("#### Original Image")
    st.image(original_image,
            use_column_width=True)

    json_tab, statistics_tab, project_tab = st.tabs(["Results & JSON Output", "Prediction Statistics", "Project Info"])

    with json_tab:
        ## Display results dataframe in main app.
        st.write('### Prediction Results (Pandas DataFrame)')
        st.dataframe(df_drawBoxes)
        ## Display the JSON in main app.
        st.write('### JSON Output')
        st.write(json_values)

    with statistics_tab:
        ## Summary statistics section in main app.
        st.write('### Summary Statistics')
        st.metric(label='Number of Bounding Boxes (ignoring overlap thresholds)', value=f"{len(df_drawBoxes.index)}")
        st.metric(label='Average Confidence Level of Bounding Boxes:', value=f"{(np.round(np.mean(df_drawBoxes['confidence'].to_numpy()),4))}")

        ## Histogram in main app.
        st.write('### Histogram of Confidence Levels')
        fig, ax = plt.subplots()
        ax.hist(df_drawBoxes['confidence'], bins=10, range=(0.0,1.0))
        st.pyplot(fig)

    with project_tab:
        st.write(f"Annotation Group Name: {project.annotation}")
        col1, col2, col3 = st.columns(3)
        for version_number in range(len(project_metadata)):
            try:
                if int(project_metadata[version_number]['model']['id'].split('/')[1]) == int(version.version):
                    col1.write(f'Total images in the version: {version.images}')
                    col1.metric(label='Augmented Train Set Image Count', value=version.splits['train'])
                    col2.metric(label='mean Average Precision (mAP)', value=f"{float(project_metadata[version_number]['model']['map'])}%")
                    col2.metric(label='Precision', value=f"{float(project_metadata[version_number]['model']['precision'])}%")
                    col2.metric(label='Recall', value=f"{float(project_metadata[version_number]['model']['recall'])}%")
                    col3.metric(label='Train Set Image Count', value=project.splits['train'])
                    col3.metric(label='Valid Set Image Count', value=project.splits['valid'])
                    col3.metric(label='Test Set Image Count', value=project.splits['test'])
            except KeyError:
                continue

        col4, col5, col6 = st.columns(3)
        col4.write('Preprocessing steps applied:')
        col4.json(version.preprocessing)
        col5.write('Augmentation steps applied:')
        col5.json(version.augmentation)
        col6.metric(label='Train Set', value=version.splits['train'], delta=f"Increased by Factor of {(version.splits['train'] / project.splits['train'])}")
        col6.metric(label='Valid Set', value=version.splits['valid'], delta="No Change")
        col6.metric(label='Test Set', value=version.splits['test'], delta="No Change")

##########
##### Set up sidebar.
##########
# Add in location to select image.
st.sidebar.write("#### Select an image to upload.")
uploaded_file = st.sidebar.file_uploader("",
                                        type=['png', 'jpg', 'jpeg'],
                                        accept_multiple_files=False)

st.sidebar.write("[Find additional images on Roboflow Universe.](https://universe.roboflow.com/)")
st.sidebar.write("[Improving Your Model with Active Learning](https://help.roboflow.com/implementing-active-learning)")

## Add in sliders.
show_bbox = st.sidebar.radio("Show Bounding Boxes:",
                            options=['Yes', 'No'],
                            key='include_bbox')

show_class_label = st.sidebar.radio("Show Class Labels:",
                                    options=['Show Labels', 'Hide Labels'],
                                    key='include_class')

show_box_type = st.sidebar.selectbox("Display Bounding Boxes As:",
                                    options=('regular', 'fill', 'blur'),
                                    key='box_type')

confidence_threshold = st.sidebar.slider("Confidence threshold (%): What is the minimum acceptable confidence level for displaying a bounding box?", 0, 100, 40, 1)
overlap_threshold = st.sidebar.slider("Overlap threshold (%): What is the maximum amount of overlap permitted between visible bounding boxes?", 0, 100, 30, 1)

image = Image.open("./images/roboflow_logo.png")
st.sidebar.image(image,
                use_column_width=True)

image = Image.open("./images/streamlit_logo.png")
st.sidebar.image(image,
                use_column_width=True)
        
##########
##### Set up project access.
##########

## Title.
st.write("# Roboflow Object Detection Tests")

with st.form("project_access"):
  workspace_id = st.text_input('Workspace ID', key='workspace_id',
                               help='Finding Your Project Information: https://docs.roboflow.com/python#finding-your-project-information-manually',
                               placeholder='Input Workspace ID')
  model_id = st.text_input('Model ID', key='model_id', placeholder='Input Model ID')
  version_number = st.text_input('Trained Model Version Number', key='version_number', placeholder='Input Trained Model Version Number')
  private_api_key = st.text_input('Private API Key', key='private_api_key', type='password',placeholder='Input Private API Key')
  submitted = st.form_submit_button("Verify and Load Model")
  if submitted:
    st.write("Loading model...")
    run_inference()