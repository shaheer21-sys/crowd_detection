######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 9/28/19
# Description:
# This program uses a TensorFlow Lite object detection model to perform object
# detection on an image or a folder full of images. It draws boxes and scores
# around the objects of interest in each image.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from shapely.geometry import box

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Parse input image name and directory.
IM_NAME = args.image
IM_DIR = args.imagedir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test8.jpg'

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Loop over every image and perform detection
for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (300, 300))
    # if image.shape[0]>800:
    #     image = image_resize(image,height=800)
    # else:
    #     pass
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

#     weightMatrix = []
# for k in range(motifWidth):
#     weightMatrix.append({'A':0,'C':0,'G':0,'T':0})
    count=0
    box_data=[]
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            box_data.append((xmin,ymin,xmax,ymax))

            print((xmin,ymin,xmax,ymax))

            # Draw label
            object_name ="person{}".format(count)
            count+=1
            #object_name = labels[int(classes[1])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text



    # All the results have been drawn on the image, now display the image
    cv2.imshow('Object detector', image)
    cv2.imwrite('output1.jpg',image)

    for box_1 in range(0,len(box_data)):
        for box_2 in range(0,len(box_data)):
            if (box_1!=box_2):

                xmin_1=box_data[box_1][0]
                ymin_1=box_data[box_1][1]
                xmax_1=box_data[box_1][2]
                ymax_1=box_data[box_1][3]

                xmin_2=box_data[box_2][0]
                ymin_2=box_data[box_2][1]
                xmax_2=box_data[box_2][2]
                ymax_2=box_data[box_2][3]

                poly_1 = box(xmin_1,ymin_1,xmax_1,ymax_1)
                poly_2 = box(xmin_2,ymin_2,xmax_2,ymax_2)

                # poly_1 = box(box_data[box_name])
                # poly_2 = box(box_data[box_name+1])
                iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

                print(iou)







        # if ( box_data[box_name]['x1'] <= box_data[box_name]['x2']
        #     or box_data[box_name]['y1'] <= box_data[box_name]['y2']
        #     or box_data[box_name+1]['x1'] <= box_data[box_name+1]['x2']
        #     or box_data[box_name+1]['y1'] <= box_data[box_name+1]['y2'] ):
        #
        #
        #      x_left = max(box_data[box_name]["x1"], box_data[box_name+1]["x1"])
        #      y_top = max(box_data[box_name]["y1"],box_data[box_name+1]["y1"])
        #      x_right = min(box_data[box_name]["x2"], box_data[box_name+1]["x2"])
        #      y_bottom = min(box_data[box_name]["y2"], box_data[box_name+1]["y2"])
        #
        #
        #      print(x_left,y_top,x_right,y_bottom)
        #
        #      if x_right < x_left or y_bottom < y_top:
        #          print("iou=0")
        #      else:
        #          # The intersection of two axis-aligned bounding boxes is always an
        #          # axis-aligned bounding box
        #           intersection_area = (x_right - x_left+1) * (y_bottom - y_top+1)
        #           print("intersection_area = ",intersection_area)
        #
        #
        #
        #           #18MayIOU
        #
        #            # compute the area of both AABBs
        #           bb1_area = (box_data[box_name]["x2"] - box_data[box_name]["x1"] +1) * (box_data[box_name]["y2"] - box_data[box_name]["y1"] +1)
        #           bb2_area = (box_data[box_name+1]["x2"] - box_data[box_name+1]["x1"] +1) * (box_data[box_name+1]["y2"] - box_data[box_name+1]["y1"] +1)
        #
        #           print(bb1_area,bb2_area)
        #              # compute the intersection over union by taking the intersection
        #              # area and dividing it by the sum of prediction + ground-truth
        #              # areas - the interesection area
        #           iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        #           print(iou)
        #           # or iou >= 0.0
        #           # or iou <= 1.0
        #           # return iou




    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(0) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation = inter)
	return resized
