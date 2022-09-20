from roboflow import Roboflow
import os, sys, re, glob
import cv2
from datetime import date
from PIL import Image, ImageDraw, ImageFont

class LocalImage():
    def __init__(self, rfModel, img_path):
        self.rfModel = rfModel
        self.img_path = img_path

    def saveImage(self, img, file_path, from_function = 'drawBoxes'):
        # helper function to save image with added text and return new image path
        if from_function == 'drawBoxes':
            filename = f'inferenceResult_{os.path.basename(file_path)}'
            cv2.imwrite(filename, img)

            return filename, img
        
        elif from_function == 'drawFilledBoxes':
            filename = f'filledBoxes_{os.path.basename(file_path)}'
            cv2.imwrite(filename, img)

            return filename, img

        elif from_function == 'blurBoxes':
            filename = f'blurredBoxes_{os.path.basename(file_path)}'
            cv2.imwrite(filename, img)

            return filename, img

        else:
            print("""Error/not the result you expected? : Please call the saveImage method with one of 'drawBoxes', 
                'drawFilledBoxes', or'blurBoxes' for the from_function keyword argument""")
            
            return

    def viewImage(self, img):
        # display resulting image
        cv2.imshow('Predicted Image', img)
        # On "q" keypress, exit
        if(cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()


    def makePrediction(self):
        if os.path.isfile(self.img_path):
            img = cv2.imread(self.img_path)
            # perform inference on the selected image
            predictions = self.rfModel.predict(self.img_path, confidence=40,
                overlap=30)
            predictions_json = predictions.json()
            print(predictions_json)
        
        elif os.path.isdir(self.img_path):
            raw_data_location = self.img_path
            for raw_data_ext in ['.jpg', '.jpeg', 'png']:
                globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
                for img_file in globbed_files:
                #   Load Image with PIL
                    img = cv2.imread(img_file)
                    predictions = self.rfModel.predict(img_file, confidence=40,
                    overlap=30)
                    predictions_json = predictions.json()
                    print(predictions_json)             
        
        else:
            return print('Please input a valid path to an existing image file or directory (folder) with images')
        
    def drawBoxes(self, printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True,
        view_img = False, include_bbox = True, include_class = True):
        self.printJson = printJson
        self.font = font
        self.save_img = save_img
        self.view_img = view_img
        self.include_bbox = include_bbox
        self.include_class = include_class

        if os.path.isfile(self.img_path):
            img = cv2.imread(self.img_path)
            # perform inference on the selected image
            predictions = self.rfModel.predict(self.img_path, confidence=40,
                overlap=30)
            predictions_json = predictions.json()
            # drawing bounding boxes with the Pillow library
            # https://docs.roboflow.com/inference/hosted-api#response-object-format
            for bounding_box in predictions:
                x0 = bounding_box['x'] - bounding_box['width'] / 2
                x1 = bounding_box['x'] + bounding_box['width'] / 2
                y0 = bounding_box['y'] - bounding_box['height'] / 2
                y1 = bounding_box['y'] + bounding_box['height'] / 2
                class_name = bounding_box['class']
                box = (x0, x1, y0, y1)
                # position coordinates: start = (x0, y0), end = (x1, y1)
                # color = RGB-value for bounding box color, (0,0,0) is "black"
                # thickness = stroke width/thickness of bounding box
                start_point = (int(x0), int(y0))
                end_point = (int(x1), int(y1))

                if self.include_bbox:
                    # draw/place bounding boxes on image
                    cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)
                
                if self.include_class:
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
 
            if save_img:
                result_path, result_img = self.saveImage(img, self.img_path, from_function='drawBoxes')
                print(f'Success! Image saved to {result_path}')

            if view_img:
                self.viewImage(img, self.img_path)

            if printJson:
                print(predictions_json)
        
        elif os.path.isdir(self.img_path):
            raw_data_location = self.img_path
            for raw_data_ext in ['.jpg', '.jpeg', 'png']:
                globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
                for img_file in globbed_files:
                #   Load Image with PIL
                    img = cv2.imread(img_file)
                    predictions = self.rfModel.predict(img_file, confidence=40,
                    overlap=30)
                    predictions_json = predictions.json()
                    # drawing bounding boxes with the Pillow library
                    # https://docs.roboflow.com/inference/hosted-api#response-object-format
                    for bounding_box in predictions:
                        x0 = bounding_box['x'] - bounding_box['width'] / 2
                        x1 = bounding_box['x'] + bounding_box['width'] / 2
                        y0 = bounding_box['y'] - bounding_box['height'] / 2
                        y1 = bounding_box['y'] + bounding_box['height'] / 2
                        class_name = bounding_box['class']
                        box = (x0, x1, y0, y1)
                        # position coordinates: start = (x0, y0), end = (x1, y1)
                        # color = RGB-value for bounding box color, (0,0,0) is "black"
                        # thickness = stroke width/thickness of bounding box
                        start_point = (int(x0), int(y0))
                        end_point = (int(x1), int(y1))

                        if self.include_bbox:
                            # draw/place bounding boxes on image
                            cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)
                        
                        if self.include_class:
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
                    
                    if save_img:
                        result_path, result_img = self.saveImage(img, img_file, from_function='drawBoxes')
                        print(f'Success! Image saved to {result_path}')

                    if view_img:
                        self.viewImage(img, img_file)

                    if printJson:
                        print(predictions_json)
        
        else:
            return print('Please input a valid path to an image or directory (folder)')

    def drawFilledBoxes(self, printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True,
        view_img = False, include_class = False):
        self.printJson = printJson
        self.font = font
        self.save_img = save_img
        self.view_img = view_img
        self.include_class = include_class

        if os.path.isfile(self.img_path):
            img = cv2.imread(self.img_path)
            # perform inference on the selected image
            predictions = self.rfModel.predict(self.img_path, confidence=40,
                overlap=30)
            predictions_json = predictions.json()
            # drawing bounding boxes with the Pillow library
            # https://docs.roboflow.com/inference/hosted-api#response-object-format
            for bounding_box in predictions:
                x0 = bounding_box['x'] - bounding_box['width'] / 2
                x1 = bounding_box['x'] + bounding_box['width'] / 2
                y0 = bounding_box['y'] - bounding_box['height'] / 2
                y1 = bounding_box['y'] + bounding_box['height'] / 2
                class_name = bounding_box['class']
                box = (x0, x1, y0, y1)
                # position coordinates: start = (x0, y0), end = (x1, y1)
                # color = RGB-value for bounding box color, (0,0,0) is "black"
                # thickness = stroke width/thickness of bounding box
                start_point = (int(x0), int(y0))
                end_point = (int(x1), int(y1))
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)
                
                if self.include_class:
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
 
            if save_img:
                result_path, result_img = self.saveImage(img, self.img_path, from_function='drawFilledBoxes')
                print(f'Success! Image saved to {result_path}')

            if view_img:
                self.viewImage(img, self.img_path)

            if printJson:
                print(predictions_json)
        
        elif os.path.isdir(self.img_path):
            raw_data_location = self.img_path
            for raw_data_ext in ['.jpg', '.jpeg', 'png']:
                globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
                for img_file in globbed_files:
                #   Load Image with PIL
                    img = cv2.imread(img_file)
                    predictions = self.rfModel.predict(img_file, confidence=40,
                    overlap=30)
                    predictions_json = predictions.json()
                    # drawing bounding boxes with the Pillow library
                    # https://docs.roboflow.com/inference/hosted-api#response-object-format
                    for bounding_box in predictions:
                        x0 = bounding_box['x'] - bounding_box['width'] / 2
                        x1 = bounding_box['x'] + bounding_box['width'] / 2
                        y0 = bounding_box['y'] - bounding_box['height'] / 2
                        y1 = bounding_box['y'] + bounding_box['height'] / 2
                        class_name = bounding_box['class']
                        box = (x0, x1, y0, y1)
                        # position coordinates: start = (x0, y0), end = (x1, y1)
                        # color = RGB-value for bounding box color, (0,0,0) is "black"
                        # thickness = stroke width/thickness of bounding box
                        start_point = (int(x0), int(y0))
                        end_point = (int(x1), int(y1))
                        # draw/place bounding boxes on image
                        cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=-1)
                        
                        if self.include_class:
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

                    if save_img:
                        result_path, result_img = self.saveImage(img, img_file, from_function='drawFilledBoxes')
                        print(f'Success! Image saved to {result_path}')

                    if view_img:
                        self.viewImage(img, img_file)

                    if printJson:
                        print(predictions_json)
        
        else:
            return print('Please input a valid path to an image or directory (folder)')

    def blurBoxes(self, printJson = True, font = cv2.FONT_HERSHEY_SIMPLEX, save_img = True,
        view_img = False, include_bbox = False, include_class = False):
        self.printJson = printJson
        self.font = font
        self.save_img = save_img
        self.view_img = view_img
        self.include_bbox = include_bbox
        self.include_class = include_class

        if os.path.isfile(self.img_path):
            img = cv2.imread(self.img_path)
            # perform inference on the selected image
            predictions = self.rfModel.predict(self.img_path, confidence=40,
                overlap=30)
            predictions_json = predictions.json()
            # drawing bounding boxes with the Pillow library
            # https://docs.roboflow.com/inference/hosted-api#response-object-format
            for bounding_box in predictions:
                x0 = bounding_box['x'] - bounding_box['width'] / 2
                x1 = bounding_box['x'] + bounding_box['width'] / 2
                y0 = bounding_box['y'] - bounding_box['height'] / 2
                y1 = bounding_box['y'] + bounding_box['height'] / 2
                class_name = bounding_box['class']
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

                # position coordinates: start = (x0, y0), end = (x1, y1)
                # color = RGB-value for bounding box color, (0,0,0) is "black"
                # thickness = stroke width/thickness of bounding box
                start_point = (int(x0), int(y0))
                end_point = (int(x1), int(y1))
                # draw/place bounding boxes on image
                cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

                if self.include_class:
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
 
            if save_img:
                result_path, result_img = self.saveImage(img, self.img_path, from_function='blurBoxes')
                print(f'Success! Image saved to {result_path}')

            if view_img:
                self.viewImage(img, self.img_path)

            if printJson:
                print(predictions_json)
        
        elif os.path.isdir(self.img_path):
            raw_data_location = self.img_path
            for raw_data_ext in ['.jpg', '.jpeg', 'png']:
                globbed_files = glob.glob(raw_data_location + '/*' + raw_data_ext)
                for img_file in globbed_files:
                #   Load Image with PIL
                    img = cv2.imread(img_file)
                    predictions = self.rfModel.predict(img_file, confidence=40,
                    overlap=30)
                    predictions_json = predictions.json()
                    # drawing bounding boxes with the Pillow library
                    # https://docs.roboflow.com/inference/hosted-api#response-object-format
                    for bounding_box in predictions:
                        x0 = bounding_box['x'] - bounding_box['width'] / 2
                        x1 = bounding_box['x'] + bounding_box['width'] / 2
                        y0 = bounding_box['y'] - bounding_box['height'] / 2
                        y1 = bounding_box['y'] + bounding_box['height'] / 2
                        class_name = bounding_box['class']
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

                        # position coordinates: start = (x0, y0), end = (x1, y1)
                        # color = RGB-value for bounding box color, (0,0,0) is "black"
                        # thickness = stroke width/thickness of bounding box
                        start_point = (int(x0), int(y0))
                        end_point = (int(x1), int(y1))
                        # draw/place bounding boxes on image
                        cv2.rectangle(img, start_point, end_point, color=(0,0,0), thickness=2)

                        if self.include_class:
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

                    if save_img:
                        result_path, result_img = self.saveImage(img, img_file, from_function='blurBoxes')
                        print(f'Success! Image saved to {result_path}')

                    if view_img:
                        self.viewImage(img, img_file)

                    if printJson:
                        print(predictions_json)
        
        else:
            return print('Please input a valid path to an image or directory (folder)')