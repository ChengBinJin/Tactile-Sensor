import os
import sys
import cv2
import scipy.misc
import numpy as np
from datetime import datetime


class RecordVideo(object):
    def __init__(self, is_record, height=480, width=640, vname=None):
        self.is_record = False
        if is_record:
            self.video_name = datetime.now().strftime("./videos/%Y%m%d-%H%M") + '-' + vname + '.avi'
            # define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.video_name, fourcc, 20.0, (width, height))
            self.is_record = True

    def turn_off(self):
        if self.is_record is True:
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
                return None, None

            rgb_frame = frame[:, :, ::-1]  # bgr to rgb
            left_img, right_img = rgb_frame[:, :self.width, :], rgb_frame[:, self.width:, :]

            return left_img, right_img


def show_stereo(imgs, args, video_writer=None, blob_writer=None):
    left_img, right_img, left_blobs, right_blobs = imgs

    window_name = 'Input Frames'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)

    # Combine two imgs, height = 480, width = 640
    h, w, ch = left_img.shape
    img = np.zeros((2*h, 2*w, ch), dtype=np.uint8)
    img[:h, :w, :], img[:h, w:2*w, :] = left_img, right_img
    img[h:2 * h, :w, :], img[h:2 * h, w:2 * w, :] = left_blobs, right_blobs

    # Display the input frame
    cv2.imshow(window_name, img)

    if args.video_record is True:
        video_writer.output.write(img[:h, :2*w, :])

    if args.result_record is True:
        blob_writer.output.write(img)


def show_disparity(stereo, args, det_results, disp_writer):
    mask, left_img, right_img = det_results['mask'], det_results['left_thres'], det_results['right_thres']

    h, w = left_img.shape
    canvas = np.zeros((2*h, 2*w), dtype=np.float32)

    real_disparity = stereo.compute(left_img, right_img)
    norm_disparity = stereo.get_norm_disparity()

    canvas[:h, :w], canvas[:h, w:2*w] = real_disparity, norm_disparity
    canvas[h:2*h, :w], canvas[h:2*h, w:2*w] = real_disparity * (mask / 255), norm_disparity * (mask / 255)

    if args.result_record is True:
        rec_canvas = create_record_canvas(h, w, real_disparity, norm_disparity, mask)
        disp_writer.output.write(rec_canvas)

    cv2.imshow('Real & Normalized Disparity', canvas)


def create_record_canvas(h, w, real_disparity, norm_disparity, mask):
    canvas = np.zeros((2*h, 2*w), dtype=np.uint8)

    real_disparity_2 = np.zeros(real_disparity.shape, dtype=np.uint8)
    real_disparity_2[real_disparity > 1.] = 255
    norm_disparity_2 = norm_disparity * 255
    norm_disparity_2[norm_disparity_2 < 0] = 0

    canvas[:h, :w], canvas[:h, w:2*w] = real_disparity_2, norm_disparity_2
    canvas[h:2*h, :w], canvas[h:2*h, w:2*w] = real_disparity_2 * (mask / 255), norm_disparity_2 * (mask / 255)

    return np.dstack((canvas, canvas, canvas))  # video write just save 3 channel frame


def all_files_under(path, extension=None, append_path=True, prefix=None, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if prefix in fname]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)
                         if (fname.endswith(extension)) and (prefix in fname)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if prefix in fname]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)
                         if (fname.endswith(extension)) and (prefix in fname)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def load_data(img_path, img_size, is_gray_scale=False, mode=0):
    img = load_image(img_path=img_path, img_size=img_size, is_gray_scale=is_gray_scale)

    if (mode == 2) or (mode == 3):
        img = binarize_img(img)
    img = img / 127.5 - 1.0

    # hope output should be [h, w, c]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return img


def binarize_img(img, is_show=False):
    threshold = 255. * 0.2
    _, thres_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    output = cv2.morphologyEx(thres_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if is_show:
        cv2.imshow('output', output)
        cv2.waitKey(0)

    return output


def load_image(img_path, img_size, is_gray_scale=False):
    if is_gray_scale:
        img = scipy.misc.imread(img_path, flatten=True).astype(np.float)
    else:
        img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)

        if not (img.ndim == 3 and img.shape[2] == 3):
            img = np.dstack((img, img, img))

    if img_size is not None:
        img = scipy.misc.imresize(img, img_size)

    return img


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()
