import sys
import cv2
import numpy as np
from datetime import datetime


class RecordVideo(object):
    def __init__(self, is_record):
        self.is_record = False
        if is_record:
            self.video_name = datetime.now().strftime("./videos/%Y%m%d-%H%M") + '.avi'
            width, height = 640, 480

            # define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.video_name, fourcc, 20.0, (width*2, height))
            self.is_record = True

    def turn_off(self):
        if self.is_record:
            self.output.release()


class Reader(object):
    def __init__(self, input_data):
        self.height, self.width = 480, 640

        if input_data == 'None':
            self.left_cap = cv2.VideoCapture(0)  # 0: left camera
            self.right_cap = cv2.VideoCapture(1)  # 1: right camera
            self.is_online = True
        else:
            self.video_cap = cv2.VideoCapture(input_data)
            self.is_online = False

    def turn_off(self):
        if self.is_online:
            self.left_cap.release()
            self.right_cap.release()
        else:
            self.video_cap.release()

    def next_frame(self):
        if self.is_online:
            left_ret, left_img = self.left_cap.read()
            right_ret, right_img = self.right_cap.read()

            if not left_ret and not right_ret:
                print('Can not read frame from one of the camera!')

            return left_img, right_img
        else:
            ret, frame = self.video_cap.read()
            if not ret:
                print('Can not read next frame from input video!')
                sys.exit()

            rgb_frame = frame[:, :, ::-1]  # bgr to rgb
            left_img, right_img = rgb_frame[:, :self.width, :], rgb_frame[:, self.width:, :]

            return left_img, right_img


def show_stereo(imgs, video_record, video_writer=None):
    left_img, right_img, left_blobs, right_blobs = imgs

    cv2.namedWindow('Stereo')
    cv2.moveWindow('Stereo', 0, 0)

    # Combine two imgs, height = 480, width = 640
    h, w, ch = left_img.shape
    img = np.zeros((2*h, 2*w, ch), dtype=np.uint8)
    img[:h, :w, :], img[:h, w:2*w, :] = left_img, right_img
    # img[h:2*h, :w, :], img[h:2*h, w:2*w, :] = np.expand_dims(left_blobs, axis=2), np.expand_dims(right_blobs, axis=2)
    img[h:2 * h, :w, :], img[h:2 * h, w:2 * w, :] = left_blobs, right_blobs

    # Display the input frame
    cv2.imshow('Stereo', img)

    if video_record is True:
        video_writer.output.write(img)


def show_disparity(stereo, mask, left_img, right_img):
    h, w = left_img.shape
    canvas = np.zeros((2*h, 2*w), dtype=np.float32)

    real_disparity = stereo.compute(left_img, right_img)
    norm_disparity = stereo.get_norm_disparity()

    canvas[:h, :w], canvas[:h, w:2*w] = real_disparity, norm_disparity
    canvas[h:2*h, :w], canvas[h:2*h, w:2*w] = real_disparity * mask, norm_disparity * mask

    cv2.imshow('Real & Normalized Disparity', canvas)
