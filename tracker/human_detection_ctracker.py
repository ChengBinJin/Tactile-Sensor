# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through
# the process of using a pre-trained model to detect objects in an image.
#  Make sure to follow the [installation instructions]
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

import numpy as np
import os
import sys
import tensorflow as tf
import time

# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from tracker.visualization import Visualization
from cTracker import cTracker
import cv2

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util


PATH_TO_CKPT = 'human_gender_detection_0614.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'human_label_map.pbtxt'
# PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'

NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory.
# create a graph to store the loaded frozen model
detection_graph = tf.Graph()
# set it as default
with detection_graph.as_default():
    # create a graph def object for protobuf to load graph from text or binary file
    # which are ready for other languages such as Java, C, Python, Go, Ruby, C#
    od_graph_def = tf.GraphDef()
    # access to the model file and read it
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        # put it into the graph def object
        od_graph_def.ParseFromString(serialized_graph)

        # import the loaded graph to the current graph in tensorflow (default graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`. # Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 5)]


visualization_ = Visualization(sequence_name="Resultss", update_ms=5)

cap = cv2.VideoCapture('014.avi')
# cap = cv2.VideoCapture(0)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # definite input and output tensors for detection graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects
        # score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        i = 0
        tracker = cTracker()

        colurs = np.random.rand(32,3)
        colours1 = np.array([255,0,0,255,0,255,255,127,127])
        colours2 = np.array([0,255,0,255,255,0,127,0,0])
        colours3 = np.array([0,0,255,0,255,255,255,255,127])

        
        
        # for _ in range(10):
        while cap.isOpened():
            # image_path = TEST_IMAGE_PATHS[1]
            i += 1
            # image = Image.open(image_path)
            # image_np = load_image_into_numpy_array(image)
            ret, image = cap.read()
            if image is None:
                break
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np = np.array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            detection_list = []
            tracking_result = []

            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            start_time = time.time()
            (boxes, scores, classes, num) = sess.run( \
                [detection_boxes, detection_scores, detection_classes, num_detections], \
                feed_dict={image_tensor: image_np_expanded})
            # print('Detection: frame %d: %.3f fps' % (i, (time.time() - start_time)))
          
            # Visualization of the results of a detection.
            image1, det_results = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)



            tracker_result = tracker.update(det_results)



            for d in tracker_result:
                d = d.astype(np.int32)
                cv2.rectangle(image1, (d[0],d[1]), (d[2],d[3]), (int(colours1[d[4]%9]),int(colours2[d[4]%9]),int(colours3[d[4]%9])), 2)
            image1 = cv2.resize(image1,(800,600))
            cv2.imshow('Result', image1)

            # print(tracker_result)

            
            

            # visualization_.set_image(image1)
            # visualization_.draw_trackers(tracker_result)
            # visualization_.show_image(image1)
            

            # image_resized = cv2.resize(image_np, (800, 600))
            # image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
            # cv2.imshow('object detection', image_resized)
            cv2.waitKey(10)