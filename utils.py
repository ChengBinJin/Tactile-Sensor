import numpy as np
import cv2
from datetime import datetime


class RecordVideo(object):
    def __init__(self, is_record):
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
                print('Can not read frame from input video!')

            rgb_frame = frame[:, :, ::-1]  # bgr to rgb
            left_img, right_img = rgb_frame[:, :self.width, :], rgb_frame[:, self.width:, :]

            return left_img, right_img


def show_stereo(left_img, right_img, video_record, video_writer=None):
    cv2.namedWindow('Stereo')
    cv2.moveWindow('Stereo', 0, 0)

    # Combine two imgs, height = 480, width = 640
    h, w, ch = left_img.shape
    img = np.zeros((h, 2 * w, ch), dtype=np.uint8)
    img[:, :w, :], img[:, w:2 * w, :] = left_img, right_img

    # Display the input frame
    cv2.imshow('Stereo', img)

    if video_record is True:
        video_writer.output.write(img)


def show_disparity(stereo, left_img, right_img):
    real_disparity = stereo.compute(left_img, right_img)
    norm_disparity = stereo.get_norm_disparity()

    cv2.imshow('real_disparity', real_disparity)
    cv2.imshow('norm_disparity', norm_disparity)
