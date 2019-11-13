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
    def __init__(self, data, mode=0, domain='xy', img_format='.jpg', resize_factor=0.5, num_attribute=6, is_train=True,
                 log_dir=None, is_debug=False):
        self.data= data
        self.mode = mode
        self.domain = domain
        self.img_format = img_format
        self.resize_factor = resize_factor
        self.is_train = is_train
        self.log_dir = log_dir
        self.is_debug = is_debug
        self.top_left = (20, 100)
        self.bottom_right = (390, 515)
        self.binarize_threshold = 55.
        self.input_shape = (int(np.ceil(self.resize_factor * (self.bottom_right[0] - self.top_left[0]))),
                            int(np.ceil(self.resize_factor * (self.bottom_right[1] - self.top_left[1]))),
                            2 if self.mode == 0 else 1)
        self.num_attribute = num_attribute
        self.small_value = 1e-7

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='rg_dataset')

        self._read_min_max_info()   # read min and max values from the .npy file
        self._read_img_path()       # read all img paths

        self.print_parameters()
        if self.is_debug and self.mode == 0:
            self._debug_roi_test()

    def print_parameters(self):
        if self.is_train:
            self.logger.info('\nDataset parameters:')
            self.logger.info('Data: \t\t\trg_{}_train_{}'.format(self.domain, self.data))
            self.logger.info('Mode: \t\t\t{}'.format(self.mode))
            self.logger.info('Domain: \t\t\t{}'.format(self.domain))
            self.logger.info('img_format: \t\t\t{}'.format(self.img_format))
            self.logger.info('resize_factor: \t\t{}'.format(self.resize_factor))
            self.logger.info('is_train: \t\t\t{}'.format(self.is_train))
            self.logger.info('is_debug: \t\t\t{}'.format(self.is_debug))
            self.logger.info('top_left: \t\t\t{}'.format(self.top_left))
            self.logger.info('bottom_right: \t\t{}'.format(self.bottom_right))
            self.logger.info('binarize_threshold: \t\t{}'.format(self.binarize_threshold))
            self.logger.info('Num. of train samples: \t{}'.format(self.num_train))
            self.logger.info('Num. of val samples: \t{}'.format(self.num_val))
            self.logger.info('Num of test samples: \t{}'.format(self.num_test))
            self.logger.info('Num. of train left_imgs: \t{}'.format(len(self.train_left_img_paths)))
            self.logger.info('Num. of train right_imgs: \t{}'.format(len(self.train_right_img_paths)))
            self.logger.info('Num. of val left_imgs: \t{}'.format(len(self.val_left_img_paths)))
            self.logger.info('Num. of val right_imgs: \t{}'.format(len(self.val_right_img_paths)))
            self.logger.info('Num. of test left_imgs: \t{}'.format(len(self.test_left_img_paths)))
            self.logger.info('Num. of test right_imgs: \t{}'.format(len(self.test_right_img_paths)))
            self.logger.info('input_shape: \t\t{}'.format(self.input_shape))
            self.logger.info('Num. of attributes: \t\t{}'.format(self.num_attribute))
            self.logger.info('Small value: \t\t{}'.format(self.small_value))
            self.logger.info('X min: \t\t\t{:.3f}'.format(self.x_min))
            self.logger.info('X max: \t\t\t{:.3f}'.format(self.x_max))
            self.logger.info('Y min: \t\t\t{:.3f}'.format(self.y_min))
            self.logger.info('Y max: \t\t\t{:.3f}'.format(self.y_max))
            self.logger.info('Ra min: \t\t\t{:.3f}'.format(self.ra_min))
            self.logger.info('Ra max: \t\t\t{:.3f}'.format(self.ra_max))
            self.logger.info('Rb min: \t\t\t{:.3f}'.format(self.rb_min))
            self.logger.info('Rb max: \t\t\t{:.3f}'.format(self.rb_max))
            self.logger.info('F min: \t\t\t{:.3f}'.format(self.f_min))
            self.logger.info('F max: \t\t\t{:.3f}'.format(self.f_max))
            self.logger.info('D min: \t\t\t{:.3f}'.format(self.d_min))
            self.logger.info('D max: \t\t\t{:.3f}'.format(self.d_max))
        else:
            print('Dataset parameters:')
            print('Data: \t\t\trg_{}_train_{}'.format(self.domain, self.data))
            print('Mode: \t\t\t{}'.format(self.mode))
            print('Domain: \t\t{}'.format(self.domain))
            print('img_format: \t\t{}'.format(self.img_format))
            print('resize_factor: \t\t{}'.format(self.resize_factor))
            print('is_train: \t\t{}'.format(self.is_train))
            print('is_debug: \t\t{}'.format(self.is_debug))
            print('top_left: \t\t{}'.format(self.top_left))
            print('bottom_right: \t\t{}'.format(self.bottom_right))
            print('binarize_threshold: \t{}'.format(self.binarize_threshold))
            print('Num. of train samples: \t{}'.format(self.num_train))
            print('Num. of val samples: \t{}'.format(self.num_val))
            print('Numo of test samples: \t{}'.format(self.num_test))
            print('Num. of train left_imgs: \t{}'.format(len(self.train_left_img_paths)))
            print('Num. of train right_imgs: \t{}'.format(len(self.train_right_img_paths)))
            print('Num. of val left_imgs: \t{}'.format(len(self.val_left_img_paths)))
            print('Num. of val right_imgs: \t{}'.format(len(self.val_right_img_paths)))
            print('Num. of test left_imgs: \t{}'.format(len(self.test_left_img_paths)))
            print('Num. of test right_imgs: \t{}'.format(len(self.test_right_img_paths)))
            print('input_shape: \t\t{}'.format(self.input_shape))
            print('Num. of attributes: \t{}'.format(self.num_attribute))
            print('Small value: \t\t{}'.format(self.small_value))
            print('X min: \t\t\t{:.3f}'.format(self.x_min))
            print('X max: \t\t\t{:.3f}'.format(self.x_max))
            print('Y min: \t\t\t{:.3f}'.format(self.y_min))
            print('Y max: \t\t\t{:.3f}'.format(self.y_max))
            print('Ra min: \t\t{:.3f}'.format(self.ra_min))
            print('Ra max: \t\t{:.3f}'.format(self.ra_max))
            print('Rb min: \t\t{:.3f}'.format(self.rb_min))
            print('Rb max: \t\t{:.3f}'.format(self.rb_max))
            print('F min: \t\t\t{:.3f}'.format(self.f_min))
            print('F max: \t\t\t{:.3f}'.format(self.f_max))
            print('D min: \t\t\t{:.3f}'.format(self.d_min))
            print('D max: \t\t\t{:.3f}'.format(self.d_max))

    def _debug_roi_test(self, batch_size=5, color=(0, 0, 255), thickness=2, save_folder='../debug'):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        indexes = np.random.random_integers(low=0, high=self.num_train, size=batch_size)

        left_img_paths = [self.train_left_img_paths[index] for index in indexes]
        right_img_paths = [self.train_right_img_paths[index] for index in indexes]

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

            # Resize img
            left_img_resize = cv2.resize(left_img_binary , None, fx=self.resize_factor, fy=self.resize_factor,
                                         interpolation=cv2.INTER_NEAREST)
            right_img_resize = cv2.resize(right_img_binary, None, fx=self.resize_factor, fy=self.resize_factor,
                                          interpolation=cv2.INTER_NEAREST)

            roi_canvas = np.hstack([left_img_roi, right_img_roi])
            crop_canvas = np.hstack([left_img_crop, right_img_crop])
            gray_canvas = np.hstack([left_img_gray, right_img_gray])
            binary_canvas = np.hstack([left_img_binary, right_img_binary])
            resize_canvas = np.hstack([left_img_resize, right_img_resize])

            # Save img
            cv2.imwrite(os.path.join(save_folder, 's1_roi_' + os.path.basename(left_path)), roi_canvas)
            cv2.imwrite(os.path.join(save_folder, 's2_crop_' + os.path.basename(left_path)), crop_canvas)
            cv2.imwrite(os.path.join(save_folder, 's3_gray_' + os.path.basename(left_path)), gray_canvas)
            cv2.imwrite(os.path.join(save_folder, 's4_binary_' + os.path.basename(left_path)), binary_canvas)
            cv2.imwrite(os.path.join(save_folder, 's5_resize_' + os.path.basename(left_path)), resize_canvas)

    def _read_min_max_info(self):
        min_max_data = np.load(os.path.join('../data', 'rg_' + self.domain + '_train_' + self.data + '.npy'))
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
        self.train_left_img_paths, self.train_right_img_paths = list(), list()
        self.val_left_img_paths, self.val_right_img_paths = list(), list()
        self.test_left_img_paths, self.test_right_img_paths = list(), list()
        self.num_train, self.num_val, self.num_test = 0, 0, 0

        if self.is_train:
            self._read_train_img_path()     # Read training img paths
            self._read_val_img_path()       # Read val img paths
        else:
            self._read_test_img_path()      # Read test img paths

    def _read_train_img_path(self):
        if self.mode == 0 or self.mode == 1:
            self.train_left_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_' + self.domain + '_train_' + self.data),
                endswith=self.img_format, condition='L_')

        if self.mode == 0 or self.mode == 2:
            self.train_right_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_' + self.domain + '_train_' + self.data),
                endswith=self.img_format, condition='R_')

        if self.mode == 0:
            assert len(self.train_left_img_paths) == len(self.train_right_img_paths)
            self.num_train = len(self.train_left_img_paths)
        elif self.mode == 1:
            self.num_train = len(self.train_left_img_paths)
        elif self.mode == 2:
            self.num_train = len(self.train_right_img_paths)
        else:
            raise NotImplementedError

    def _read_val_img_path(self):
        if self.mode == 0 or self.mode == 1:
            self.val_left_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_' + self.domain + '_val_' + self.data),
                endswith=self.img_format, condition='L_')

        if self.mode == 0 or self.mode == 2:
            self.val_right_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_'  + self.domain + '_val_' + self.data),
                endswith=self.img_format, condition='R_')

        if self.mode == 0:
            assert len(self.val_left_img_paths) == len(self.val_right_img_paths)
            self.num_val = len(self.val_left_img_paths)
        elif self.mode == 1:
            self.num_val = len(self.val_left_img_paths)
        elif self.mode == 2:
            self.num_val = len(self.val_right_img_paths)
        else:
            raise NotImplementedError

    def _read_test_img_path(self):
        if self.mode == 0 or self.mode == 1:
            self.test_left_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_' + self.domain + '_test_' + self.data),
                endswith=self.img_format, condition='L_')

        if self.mode == 0 or self.mode == 2:
            self.test_right_img_paths = utils.all_files_under(
                folder=os.path.join('../data', 'rg_' + self.domain + '_test_' + self.data),
                endswith=self.img_format, condition='R_')

        if self.mode == 0:
            assert len(self.test_left_img_paths) == len(self.test_right_img_paths)
            self.num_test = len(self.test_left_img_paths)
        elif self.mode == 1:
            self.num_test = len(self.test_left_img_paths)
        elif self.mode == 2:
            self.num_test = len(self.test_right_img_paths)
        else:
            raise NotImplementedError

    def train_random_batch(self, batch_size=4):
        indexes = np.random.random_integers(low=0, high=self.num_train-1, size=batch_size)
        left_img_paths, right_img_paths = None, None

        if self.mode == 0 or self.mode == 1:
            left_img_paths = [self.train_left_img_paths[index] for index in indexes]

        if self.mode == 0 or self.mode == 2:
            right_img_paths = [self.train_right_img_paths[index] for index in indexes]

        return self.data_reader(left_img_paths, right_img_paths)

    def direct_batch(self, batch_size, start_index, stage='val'):
        if stage == 'val':
            num_imgs = self.num_val
            left_img_paths = self.val_left_img_paths
            right_img_paths = self.val_right_img_paths
        elif stage == 'test':
            num_imgs = self.num_test
            left_img_paths = self.test_left_img_paths
            right_img_paths = self.test_right_img_paths
        else:
            raise NotImplementedError

        if start_index + batch_size < num_imgs:
            end_index = start_index + batch_size
        else:
            end_index = num_imgs

        # Select indexes
        indexes = [idx for idx in range(start_index, end_index)]

        left_paths, right_paths = None, None
        if self.mode == 0 or self.mode == 1:
            left_paths = [left_img_paths[index] for index in indexes]
        if self.mode == 0 or self.mode == 2:
            right_paths = [right_img_paths[index] for index in indexes]

        return self.data_reader(left_paths, right_paths)

    def data_reader(self, left_img_paths, right_img_paths):
        if self.mode == 0 or self.mode == 1:
            batch_size = len(left_img_paths)
        else:
            batch_size = len(right_img_paths)
        batch_imgs = np.zeros((batch_size, *self.input_shape), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.num_attribute), dtype=np.float32)

        left_imgs = list()
        if self.mode == 0 or self.mode == 1:
            for i, left_img_path in enumerate(left_img_paths):
                left_imgs.append(self.data_preprocessing(left_img_path))
                # Process labels
                batch_labels[i, :] = utils.read_label(left_img_path, img_format=self.img_format)

        right_imgs = list()
        if self.mode == 0 or self.mode == 2:
            for i, right_img_path in enumerate(right_img_paths):
                right_imgs.append(self.data_preprocessing(right_img_path))
                # Process labels
                batch_labels[i, :] = utils.read_label(right_img_path, img_format=self.img_format)

        # Normalize labels
        batch_labels = self.normalize(batch_labels)

        if self.mode == 0:
            for i in range(len(left_imgs)):
                left_img = left_imgs[i]
                right_img = right_imgs[i]
                batch_imgs[i] = np.dstack([left_img, right_img])
        elif self.mode == 1:
            for i in range(len(left_imgs)):
                batch_imgs[i] = np.expand_dims(left_imgs[i], axis=-1)
        else:
            for i in range(len(right_imgs)):
                batch_imgs[i] = np.expand_dims(right_imgs[i], axis=-1)

        return batch_imgs, batch_labels

    def data_preprocessing(self, img_path):
        # Process imgs
        img = cv2.imread(img_path)
        # Stage 1: cropping
        img_crop = img[self.top_left[0]:self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]
        # Stage 2: BGR to Gray
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        # Stage 3: Thresholding
        _, img_binary = cv2.threshold(img_gray, self.binarize_threshold, 255., cv2.THRESH_BINARY)
        # Stage 4: Resize img
        img_resize = cv2.resize(img_binary, None, fx=self.resize_factor, fy=self.resize_factor,
                                     interpolation=cv2.INTER_NEAREST)
        return img_resize

    def normalize(self, data):
        return (data - self.min_values) / (self.max_values - self.min_values + self.small_value)

    def unnormalize(self, data):
        return data * (self.max_values - self.min_values + self.small_value) + self.min_values