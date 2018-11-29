"""camera_tf_trt.py

This is a Camera TensorFlow/TensorRT Object Detection sample code for
Jetson TX2 or TX1.  This script captures and displays video from either
a video file, an image file, an IP CAM, a USB webcam, or the Tegra
onboard camera, and do real-time object detection with example TensorRT
optimized SSD models in NVIDIA's 'tf_trt_models' repository.  Refer to
README.md inside this repository for more information.

This code is written and maintained by JK Jung <jkjung13@gmail.com>.
"""


import sys
import time
import logging
import argparse

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from utils.camera import Camera
from utils.od_utils import read_label_map, build_trt_pb, load_trt_pb, \
                           write_graph_tensorboard, detect
from utils.visualization import BBoxVisualization

import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

# Constants
DEFAULT_MODEL = 'ssd_inception_v2_coco'
DEFAULT_LABELMAP = 'third_party/models/research/object_detection/' \
                   'data/mscoco_label_map.pbtxt'
WINDOW_NAME = 'CameraTFTRTDemo'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    """Parse input arguments."""
    desc = ('This script captures and displays live camera video, '
            'and does real-time object detection with TF-TRT model '
            'on Jetson TX2/TX1')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file', dest='use_file',
                        help='use a video file as input (remember to '
                        'also set --filename)',
                        action='store_true')
    parser.add_argument('--image', dest='use_image',
                        help='use an image file as input (remember to '
                        'also set --filename)',
                        action='store_true')
    parser.add_argument('--filename', dest='filename',
                        help='video file name, e.g. test.mp4',
                        default=None, type=str)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    parser.add_argument('--model', dest='model',
                        help='tf-trt object detecion model '
                        '[{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_MODEL, type=str)
    parser.add_argument('--build', dest='do_build',
                        help='re-build TRT pb file (instead of using'
                        'the previously built version)',
                        action='store_true')
    parser.add_argument('--tensorboard', dest='do_tensorboard',
                        help='write optimized graph summary to TensorBoard',
                        action='store_true')
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format(DEFAULT_LABELMAP),
                        default=DEFAULT_LABELMAP, type=str)
    parser.add_argument('--num-classes', dest='num_classes',
                        help='(deprecated and not used) number of object '
                        'classes', type=int)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.3, type=float)
    args = parser.parse_args()
    return args


def open_display_window(width, height):
    """Open the cv2 window for displaying images with bounding boxeses."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera TFTRT Object Detection Demo '
                                    'for Jetson TX2/TX1')


def draw_help_and_fps(img, fps):
    """Draw help message and fps number at top-left corner of the image."""
    help_text = "'Esc' to Quit, 'H' for FPS & Help, 'F' for Fullscreen"
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA

    fps_text = 'FPS: {:.1f}'.format(fps)
    cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    cv2.putText(img, fps_text, (11, 50), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 50), font, 1.0, (240, 240, 240), 1, line)
    return img


def set_full_screen(full_scrn):
    """Set display window to full screen or not."""
    prop = cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)


def show_bounding_boxes(img, box, conf, cls, cls_dict):
    """Draw detected bounding boxes on the original image."""
    font = cv2.FONT_HERSHEY_DUPLEX
    for bb, cf, cl in zip(box, conf, cls):
        cl = int(cl)
        y_min, x_min, y_max, x_max = bb[0], bb[1], bb[2], bb[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BBOX_COLOR, 2)
        txt_loc = (max(x_min, 5), max(y_min-3, 20))
        cls_name = cls_dict.get(cl, 'CLASS{}'.format(cl))
        txt = '{} {:.2f}'.format(cls_name, cf)
        cv2.putText(img, txt, txt_loc, font, 0.8, BBOX_COLOR, 1)
    return img

def findLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    
    # the canny function is magical. see https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 250  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    # array element is structed as x1,y1,x2,y2
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    return lines

def loop_and_detect(cam, tf_sess, conf_th, vis, od_type):
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    show_fps = True
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while cam.thread_running:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the display window.
            # If yes, terminate the while loop.
            break

        img = cam.read()
        if img is not None:
            box, conf, cls = detect(img, tf_sess, conf_th, od_type=od_type)
            (img, x_min, y_min, x_max, y_max, cf) = vis.draw_bboxes(img, box, conf, cls)

            # Draw the court boundaries
            height, width, channels = img.shape
            COURT_WIDTH=int(width/2)
            COURT_HEIGHT=int(height/2)
            (a,b,c,d) = (int(0),int(0),int(COURT_WIDTH),int(COURT_HEIGHT))
            (e,f,g,h) = (int(COURT_WIDTH),int(0),int(2*COURT_WIDTH),int(COURT_HEIGHT))
            black_img = np.copy(img) * 0
            cv2.rectangle(black_img, (a, b), (c, d), (0,0,255), -1)
            cv2.rectangle(black_img, (e, f), (g, h), (255,0,0), -1)
            cv2.putText(img,'court 1',(30,80), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(img,'court 2',((COURT_WIDTH+30),80), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(img,'out of bounds', (30,(COURT_HEIGHT+30)), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            img = cv2.addWeighted(img,0.8,black_img,0.4,0)
            
            if conf is None:
                continue
            if (cf >= 0.6):
                if (isIntersect(a,b,c,d,x_min,y_min,x_max,y_max) == True and \
                    (isIntersect(e,f,g,h,x_min,y_min,x_max,y_max) == True)):
                    if (abs(x_min+x_max)/2 <= (COURT_WIDTH)):
                        print("left")
                    else:
                        print("right")
                elif (isIntersect(a,b,c,d,x_min,y_min,x_max,y_max) == True):
                    print("left")
                elif (isIntersect(e,f,g,h,x_min,y_min,x_max,y_max) == True):
                    print("right")
                elif (isInbound(a,b,c,d,x_min,y_min,x_max,y_max) == True):
                    print("left")
                elif (isInbound(e,f,g,h,x_min,y_min,x_max,y_max) == True):
                    print("right")
                else:
                    print("out of bounds")

            if show_fps:
                img = draw_help_and_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
            tic = toc

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help/fps
            show_fps = not show_fps
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_full_screen(full_scrn)

# https://stackoverflow.com/questions/1585525/how-to-find-the-intersection-point-between-a-line-and-a-rectangle
# rectDims takes x1,y1,x2,y2
# lines is an array of array
# output: true if the rectangle dimensions intersect with a line
def isIntersect(x1,y1,x2,y2, minX, minY, maxX, maxY):

    if ((x1 <= minX and x2 <= minX) or \
        (y1 <= minY and y2 <= minY) or \
        (x1 >= maxX and x2 >= maxX) or \
        (y1 >= maxY and y2 >= maxY)):
        return False
    
    if (y1 > minY and y1 < maxY): 
        return True
    if (y2 > minY and y2 < maxY): 
        return True
    if (x1 > minX and x1 < maxX):
        return True
    if (x2 > minX and x2 < maxX):
        return True
    
    return False

# This function finds the boundaries of a badminton court
def findBoundingBox(lines, width, height):
    minX = width
    minY = height
    maxX = 0
    maxY = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x1 < minX):
                minX = x1
            if (x2 < minX):
                minX = x2
            if (y1 < minY):
                minY = y1
            if (y2 < minY):
                minY = y2

            if (x1 > maxX):
                maxX = x1
            if (x2 > maxX):
                maxX = x2
            if (y1 > maxY):
                maxY = y1
            if (y2 > maxY):
                maxY = y2

    return (minX, minY, maxX, maxY)

# This function determines whether the birdie is in or out of bounds
def isInbound(cMinX, cMinY, cMaxX, cMaxY, bMinX, bMinY, bMaxX, bMaxY):
    if ((cMinX <= bMinX <= cMaxX) and \
        (cMinY <= bMinY <= cMaxY) and \
        (cMinX <= bMaxX <= cMaxX) and \
        (cMinY <= bMaxY <= cMaxY)):
        return True
    
    if (isIntersect(cMinX,cMinY,cMinX,cMaxY,bMinX,bMinY,bMaxX,bMaxY) or \
            isIntersect(cMaxX,cMinY,cMaxX,cMaxY,bMinX,bMinY,bMaxX,bMaxY) or \
            isIntersect(cMinX,cMinY,cMinX,cMaxY,bMinX,bMinY,bMaxX,bMaxY) or \
            isIntersect(cMaxX,cMinY,cMaxX,cMaxY,bMinX,bMinY,bMaxX,bMaxY)):
        return True

    return False


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Ask tensorflow logger not to propagate logs to parent (which causes
    # duplicated logging)
    logging.getLogger('tensorflow').propagate = False

    args = parse_args()
    logger.info('called with args: %s' % args)

    # build the class (index/name) dictionary from labelmap file
    logger.info('reading label map')
    cls_dict = read_label_map(args.labelmap_file)

    pb_path = './data/{}_trt.pb'.format(args.model)
    log_path = './logs/{}_trt'.format(args.model)
    if args.do_build:
        logger.info('building TRT graph and saving to pb: %s' % pb_path)
        build_trt_pb(args.model, pb_path)

    logger.info('opening camera device/file')
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)

    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    if args.do_tensorboard:
        logger.info('writing graph summary to TensorBoard')
        write_graph_tensorboard(tf_sess, log_path)

    logger.info('warming up the TRT graph with a dummy image')
    od_type = 'faster_rcnn' if 'faster_rcnn' in args.model else 'ssd'
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, _, _ = detect(dummy_img, tf_sess, conf_th=.3, od_type=od_type)

    cam.start()  # ask the camera to start grabbing images

    # grab image and do object detection (until stopped by user)
    logger.info('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)
    open_display_window(cam.img_width, cam.img_height)
    loop_and_detect(cam, tf_sess, args.conf_th, vis, od_type=od_type)

    logger.info('cleaning up')
    cam.stop()  # terminate the sub-thread in camera
    tf_sess.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
