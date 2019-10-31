# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import logging
import numpy as np
import utils as utils


class Dataset(object):
    def __init__(self, data, mode=0, img_format='.png', resize_factor=0.5, num_attribute=6, is_train=True,
                 log_dir=None, is_debug=False):
        self.data= data
        self.mode = mode
        self.img_format = img_format
        self.resize_factor = resize_factor
        self.is_train = is_train
        self.log_dir = log_dir
        self.is_debug = is_debug
        self.top_left = (20, 90)
        self.bottom_right = (390, 575)
        self.binarize_threshold = 20.
        self.input_shape = (int(np.floor(self.resize_factor * (self.bottom_right[0] - self.top_left[0]))),
                            int(np.floor(self.resize_factor * (self.bottom_right[1] - self.top_left[1]))), 2)
        self.num_atrribute = num_attribute
        self.small_value = 1e-7

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='rg_dataset')

        self._read_min_max_info()   # read min and max values from the .npy file
        self._read_img_path()       # read all img paths

        self.print_parameters()
        if self.is_debug:
            self._debug_roi_test()

    def print_parameters(self):
        if self.is_train:
            self.logger.info('\nDataset parameters:')
            self.logger.info('Data: \t\trg_train{}'.format(self.data))
            self.logger.info('Mode: \t\t{}'.format(self.mode))
            self.logger.info('img_format: \t\t{}'.format(self.img_format))
            self.logger.info('resize_factor: \t{}'.format(self.resize_factor))
            self.logger.info('is_train: \t\t{}'.format(self.is_train))
            self.logger.info('is_debug: \t\t{}'.format(self.is_debug))
            self.logger.info('top_left: \t\t{}'.format(self.top_left))
            self.logger.info('bottom_right: \t{}'.format(self.bottom_right))
            self.logger.info('binarize_threshold: \t{}'.format(self.binarize_threshold))
            self.logger.info('Num. of imgs: \t{}'.format(self.num_train))
            self.logger.info('input_shape: \t{}'.format(self.input_shape))
            self.logger.info('Num. of attributes: \t{}'.format(self.num_atrribute))
            self.logger.info('Small value: \t{}'.format(self.small_value))
            self.logger.info('X min: \t\t{:.3f}'.format(self.x_min))
            self.logger.info('X max: \t\t{:.3f}'.format(self.x_max))
            self.logger.info('Y min: \t\t{:.3f}'.format(self.y_min))
            self.logger.info('Y max: \t\t{:.3f}'.format(self.y_max))
            self.logger.info('Ra min: \t\t{:.3f}'.format(self.ra_min))
            self.logger.info('Ra max: \t\t{:.3f}'.format(self.ra_max))
            self.logger.info('Rb min: \t\t{:.3f}'.format(self.rb_min))
            self.logger.info('Rb max: \t\t{:.3f}'.format(self.rb_max))
            self.logger.info('F min: \t\t{:.3f}'.format(self.f_min))
            self.logger.info('F max: \t\t{:.3f}'.format(self.f_max))
            self.logger.info('D min: \t\t{:.3f}'.format(self.d_min))
            self.logger.info('D max: \t\t{:.3f}'.format(self.d_max))
        else:
            print('Dataset parameters:')
            print('Data: \t\trg_train{}'.format(self.data))
            print('Mode: \t\t{}'.format(self.mode))
            print('img_format: \t\t{}'.format(self.img_format))
            print('resize_factor: \t{}'.format(self.resize_factor))
            print('is_train: \t\t{}'.format(self.is_train))
            print('is_debug: \t\t{}'.format(self.is_debug))
            print('top_left: \t\t{}'.format(self.top_left))
            print('bottom_right: \t{}'.format(self.bottom_right))
            print('binarize_threshold: \t{}'.format(self.binarize_threshold))
            print('Num. of imgs: \t{}'.format(self.num_train))
            print('input_shape: \t{}'.format(self.input_shape))
            print('Num. of attributes: \t{}'.format(self.num_atrribute))
            print('Small value: \t{}'.format(self.small_value))
            print('X min: \t\t{:.3f}'.format(self.x_min))
            print('X max: \t\t{:.3f}'.format(self.x_max))
            print('Y min: \t\t{:.3f}'.format(self.y_min))
            print('Y max: \t\t{:.3f}'.format(self.y_max))
            print('Ra min: \t\t{:.3f}'.format(self.ra_min))
            print('Ra max: \t\t{:.3f}'.format(self.ra_max))
            print('Rb min: \t\t{:.3f}'.format(self.rb_min))
            print('Rb max: \t\t{:.3f}'.format(self.rb_max))
            print('F min: \t\t{:.3f}'.format(self.f_min))
            print('F max: \t\t{:.3f}'.format(self.f_max))
            print('D min: \t\t{:.3f}'.format(self.d_min))
            print('D max: \t\t{:.3f}'.format(self.d_max))

    def _debug_roi_test(self, batch_size=8, color=(0, 0, 255), thickness=2, save_folder='../debug'):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        indexes = np.random.random_integers(low=0, high=self.num_train, size=batch_size)

        left_img_paths = [self.left_img_paths[index] for index in indexes]
        right_img_paths = [self.right_img_paths[index] for index in indexes]

        for left_path, right_path in zip(left_img_paths, right_img_paths):
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            # Draw roi
            left_img_roi = cv2.rectangle(left_img.copy(), (self.top_left[1], self.top_left[0]),
                                         (self.bottom_right[1], self.bottom_right[0]), color=color, thickness=thickness)
            right_img_roi = cv2.rectangle(right_img.copy(), (self.top_left[1], self.top_left[0]),
                                          (self.bottom_right[1], self.bottom_right[0]), color=color, thickness=thickness)

            # Cropping
            left_img_crop = left_img[self.top_left[0]:self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]
            right_img_crop = right_img[self.top_left[0]:self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]

            # BGR to Gray
            left_img_gray = cv2.cvtColor(left_img_crop, cv2.COLOR_BGR2GRAY)
            right_img_gray = cv2.cvtColor(right_img_crop, cv2.COLOR_BGR2GRAY)

            # Thresholding
            _, left_img_binary = cv2.threshold(left_img_gray, self.binarize_threshold , 255., cv2.THRESH_BINARY)
            _, right_img_binary = cv2.threshold(right_img_gray, self.binarize_threshold , 255., cv2.THRESH_BINARY)

            # # Closing: dilation and erosion
            # kernel = np.ones((3, 3), np.float32)
            # left_img_close = cv2.morphologyEx(left_img_binary, cv2.MORPH_CLOSE, kernel)
            # right_img_close = cv2.morphologyEx(right_img_binary, cv2.MORPH_CLOSE, kernel)
            #
            # # Opening: erosion and dilation
            # kernel = np.ones((5, 5), np.float32)
            # left_img_open = cv2.morphologyEx(left_img_close, cv2.MORPH_OPEN, kernel)
            # right_img_open = cv2.morphologyEx(right_img_close, cv2.MORPH_OPEN, kernel)

            # Resize img
            left_img_resize = cv2.resize(left_img_binary , None, fx=self.resize_factor, fy=self.resize_factor)
            right_img_resize = cv2.resize(right_img_binary, None, fx=self.resize_factor, fy=self.resize_factor)

            roi_canvas = np.hstack([left_img_roi, right_img_roi])
            crop_canvas = np.hstack([left_img_crop, right_img_crop])
            gray_canvas = np.hstack([left_img_gray, right_img_gray])
            binary_canvas = np.hstack([left_img_binary, right_img_binary])
            # close_canvas = np.hstack([left_img_close, right_img_close])
            # open_canvas = np.hstack([left_img_open, right_img_open])
            resize_canvas = np.hstack([left_img_resize, right_img_resize])

            # Save img
            cv2.imwrite(os.path.join(save_folder, 's1_roi_' + os.path.basename(left_path)), roi_canvas)
            cv2.imwrite(os.path.join(save_folder, 's2_crop_' + os.path.basename(left_path)), crop_canvas)
            cv2.imwrite(os.path.join(save_folder, 's3_gray_' + os.path.basename(left_path)), gray_canvas)
            cv2.imwrite(os.path.join(save_folder, 's4_binary_' + os.path.basename(left_path)), binary_canvas)
            # cv2.imwrite(os.path.join(save_folder, 's5_close_' + os.path.basename(left_path)), close_canvas)
            # cv2.imwrite(os.path.join(save_folder, 's6_open_' + os.path.basename(left_path)), open_canvas)
            cv2.imwrite(os.path.join(save_folder, 's5_resize_' + os.path.basename(left_path)), resize_canvas)

    def _read_min_max_info(self):
        min_max_data = np.load(os.path.join('../data', 'rg_train' + self.data + '.npy'))
        self.x_min = min_max_data[0]
        self.x_max = min_max_data[1]
        self.y_min = min_max_data[2]
        self.y_max = min_max_data[3]
        self.ra_min = min_max_data[4]
        self.ra_max = min_max_data[5]
        self.rb_min = min_max_data[6]
        self.rb_max = min_max_data[7]
        self.f_min = min_max_data[8]
        self.f_max = min_max_data[9]
        self.d_min = min_max_data[10]
        self.d_max = min_max_data[11]

        self.min_values = np.asarray([self.x_min, self.y_min, self.ra_min, self.rb_min, self.f_min, self.d_min])
        self.max_values = np.asarray([self.x_max, self.y_max, self.ra_max, self.rb_max, self.f_max, self.d_max])

    def _read_img_path(self):
        self.left_img_paths = utils.all_files_under(folder=os.path.join('../data', 'rg_train' + self.data),
                                                    endswith=self.img_format,
                                                    condition='L_')
        self.right_img_paths = utils.all_files_under(folder=os.path.join('../data', 'rg_train' + self.data),
                                                     endswith=self.img_format,
                                                     condition='R_')

        assert len(self.left_img_paths) == len(self.right_img_paths)
        self.num_train = len(self.left_img_paths)

    def train_random_batch(self, batch_size=4):
        batch_imgs = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.num_atrribute), dtype=np.float32)

        indexes = np.random.random_integers(low=0, high=self.num_train-1, size=batch_size)

        left_img_paths = [self.left_img_paths[index] for index in indexes]
        right_img_paths = [self.right_img_paths[index] for index in indexes]

        for i, (left_path, right_path) in enumerate(zip(left_img_paths, right_img_paths)):
            # Process imgs
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)

            # Stage 1: cropping
            left_img_crop = left_img[self.top_left[0]:self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]
            right_img_crop = right_img[self.top_left[0]:self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]

            # Stage 2: BGR to Gray
            left_img_gray = cv2.cvtColor(left_img_crop, cv2.COLOR_BGR2GRAY)
            right_img_gray = cv2.cvtColor(right_img_crop, cv2.COLOR_BGR2GRAY)

            # Stage 3: Thresholding
            _, left_img_binary = cv2.threshold(left_img_gray, self.binarize_threshold , 255., cv2.THRESH_BINARY)
            _, right_img_binary = cv2.threshold(right_img_gray, self.binarize_threshold , 255., cv2.THRESH_BINARY)

            # Stage 4: Resize img
            left_img_resize = cv2.resize(left_img_binary, None, fx=self.resize_factor, fy=self.resize_factor)
            right_img_resize = cv2.resize(right_img_binary, None, fx=self.resize_factor, fy=self.resize_factor)

            batch_imgs[i, :] = np.dstack([left_img_resize, right_img_resize])

            # Process labels
            batch_labels[i, :] = utils.read_label(left_path)

        # Normalize labels
        batch_labels = self.normalize(batch_labels)

        return batch_imgs, batch_labels

    def normalize(self, data):
        return (data - self.min_values) / (self.max_values - self.min_values + self.small_value)

    def unnormalize(self, data):
        return data * (self.max_values - self.min_values + self.small_value) + self.min_values