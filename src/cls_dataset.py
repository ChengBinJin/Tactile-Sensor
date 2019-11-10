# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import logging
import numpy as np

import utils as utils


class Dataset(object):
    def __init__(self, shape, mode=0, img_format='.png', resize_factor=0.5, is_train=True, log_dir=None, is_debug=False):
        self.shape = shape
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
        self.train_ratio, self.val_ratio, self.test_ratio = 0.6, 0.2, 0.2

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=log_dir, is_train=is_train, name='cls_dataset')

        self._read_img_path()  # read all img paths
        self.print_parameters()


    def _read_img_path(self):
        self.cls01_left = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_1'),
                                                endswith=self.img_format,
                                                condition='_L_')
        self.cls01_right = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_1'),
                                                 endswith=self.img_format,
                                                 condition='_R_')

        self.cls02_left = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_2'),
                                           endswith=self.img_format,
                                           condition='_L_')
        self.cls02_right = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_2'),
                                           endswith=self.img_format,
                                           condition='_R_')

        self.cls03_left = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_3'),
                                           endswith=self.img_format,
                                           condition='_L_')
        self.cls03_right = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_3'),
                                                 endswith=self.img_format,
                                                 condition='_R_')

        self.cls04_left = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_4'),
                                                endswith=self.img_format,
                                                condition='_L_')
        self.cls04_right = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_4'),
                                                 endswith=self.img_format,
                                                 condition='_R_')

        self.cls05_left = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_5'),
                                                endswith=self.img_format,
                                                condition='_L_')
        self.cls05_right = utils.all_files_under(folder=os.path.join('../data', 'cls_' + self.shape, self.shape + '_5'),
                                                 endswith=self.img_format,
                                                 condition='_R_')

        self._read_train_img_path()     # Read training img paths
        self._read_val_img_path()       # Read val img paths
        self._read_test_img_path()      # Read test img paths


    def _read_train_img_path(self):
        self.train_left_img_paths = list()
        self.train_right_img_paths = list()

        left_paths = [self.cls01_left, self.cls02_left, self.cls03_left, self.cls04_left, self.cls05_left]
        right_paths = [self.cls01_right, self.cls02_right, self.cls03_right, self.cls04_right, self.cls05_right]

        for left_path, right_path in zip(left_paths, right_paths):
            self.train_left_img_paths.extend(left_path[0:int(self.train_ratio * len(left_path))])
            self.train_right_img_paths.extend(right_path[0:int(self.train_ratio * len(right_path))])

        assert len(self.train_left_img_paths) == len(self.train_right_img_paths)
        self.num_train = len(self.train_left_img_paths)


    def _read_val_img_path(self):
        self.val_left_img_paths = list()
        self.val_right_img_paths = list()

        left_paths = [self.cls01_left, self.cls02_left, self.cls03_left, self.cls04_left, self.cls05_left]
        right_paths = [self.cls01_right, self.cls02_right, self.cls03_right, self.cls04_right, self.cls05_right]

        for left_path, right_path in zip(left_paths, right_paths):
            self.val_left_img_paths.extend(left_path[int(self.train_ratio * len(left_path))
                                                     :int((self.train_ratio + self.val_ratio) * len(left_path))])
            self.val_right_img_paths.extend(right_path[int(self.train_ratio * len(left_path))
                                                       :int((self.train_ratio + self.val_ratio) * len(right_path))])

        assert len(self.val_left_img_paths) == len(self.val_right_img_paths)
        self.num_val = len(self.val_left_img_paths)


    def _read_test_img_path(self):
        self.test_left_img_paths = list()
        self.test_right_img_paths = list()

        left_paths = [self.cls01_left, self.cls02_left, self.cls03_left, self.cls04_left, self.cls05_left]
        right_paths = [self.cls01_right, self.cls02_right, self.cls03_right, self.cls04_right, self.cls05_right]

        for left_path, right_path in zip(left_paths, right_paths):
            self.test_left_img_paths.extend(left_path[-int(self.test_ratio * len(left_path)):])
            self.test_right_img_paths.extend(right_path[-int(self.test_ratio * len(right_path)):])

        assert len(self.test_left_img_paths) == len(self.test_right_img_paths)
        self.num_test = len(self.test_left_img_paths)


    def print_parameters(self):
        if self.is_train:
            self.logger.info('\nDataset parameters:')
            self.logger.info('Shape: \t\trg_train{}'.format(self.shape))
            self.logger.info('Mode: \t\t{}'.format(self.mode))
            self.logger.info('img_format: \t\t{}'.format(self.img_format))
            self.logger.info('resize_factor: \t{}'.format(self.resize_factor))
            self.logger.info('is_train: \t\t{}'.format(self.is_train))
            self.logger.info('is_debug: \t\t{}'.format(self.is_debug))
            self.logger.info('top_left: \t\t{}'.format(self.top_left))
            self.logger.info('bottom_right: \t{}'.format(self.bottom_right))
            self.logger.info('binarize_threshold: \t{}'.format(self.binarize_threshold))
            self.logger.info('Num. of train imgs: \t{}'.format(self.num_train))
            self.logger.info('Num. of val imgs: \t{}'.format(self.num_val))
            self.logger.info('Numo of test imgs: \t{}'.format(self.num_test))
            self.logger.info('input_shape: \t{}'.format(self.input_shape))
        else:
            print('Dataset parameters:')
            print('Shape: \t\trg_train{}'.format(self.shape))
            print('Data: \t\trg_train{}'.format(self.shape))
            print('Mode: \t\t{}'.format(self.mode))
            print('img_format: \t\t{}'.format(self.img_format))
            print('resize_factor: \t{}'.format(self.resize_factor))
            print('is_train: \t\t{}'.format(self.is_train))
            print('is_debug: \t\t{}'.format(self.is_debug))
            print('top_left: \t\t{}'.format(self.top_left))
            print('bottom_right: \t{}'.format(self.bottom_right))
            print('binarize_threshold: \t{}'.format(self.binarize_threshold))
            print('Num. of train imgs: \t{}'.format(self.num_train))
            print('Num. of val imgs: \t{}'.format(self.num_val))
            print('Numo of test imgs: \t{}'.format(self.num_test))
            print('input_shape: \t{}'.format(self.input_shape))

