import numpy as np
import cv2
import argparse
from datetime import datetime

from sgbm import SGBM

parser = argparse.ArgumentParser(description='')
parser.add_argument('--video_record', dest='video_record', action='store_true', default=False,
                    help='record raw video or not')
args = parser.parse_args()


class RecordVideo(object):
    def __init__(self):
        self.video_name = datetime.now().strftime("%Y%m%d-%H%M") + '.avi'
        width, height = 640, 480

        # define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output = cv2.VideoWriter(self.video_name, fourcc, 20.0, (width*2, height))


def main():
    stereo = SGBM([480, 640])

    left_cap = cv2.VideoCapture(0)   # 0: left camera
    right_cap = cv2.VideoCapture(1)  # 1: right camera

    video_writer = None
    if args.video_record is True:
        video_writer = RecordVideo()

    while True:
        # Capture frame-by-frame
        left_ret, left_img = left_cap.read()
        right_ret, right_img = right_cap.read()

        if not left_ret and not right_ret:
            print('can not read frame from one of the camera')

        show_stereo(left_img, right_img, video_writer)
        show_disparity(stereo, left_img, right_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # When everyting done, release the capture
    left_cap.release()
    right_cap.release()
    if args.video_record is True:
        video_writer.output.release()

    cv2.destroyAllWindows()


def show_stereo(left_img, right_img, video_writer=None):
    cv2.namedWindow('Stereo')
    cv2.moveWindow('Stereo', 0, 0)

    # Combine two imgs, height = 480, width = 640
    h, w, ch = left_img.shape
    img = np.zeros((h, 2 * w, ch), dtype=np.uint8)
    img[:, :w, :], img[:, w:2 * w, :] = left_img, right_img

    # Display the input frame
    cv2.imshow('Stereo', img)

    if args.video_record is True:
        video_writer.output.write(img)


def show_disparity(stereo, left_img, right_img):
    real_disparity = stereo.compute(left_img, right_img)
    norm_disparity = stereo.get_norm_disparity()

    cv2.imshow('real_disparity', real_disparity)
    cv2.imshow('norm_disparity', norm_disparity)


if __name__ == '__main__':
    main()
