# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import utils as utils


class Dataset(object):
    def __init__(self, data, is_train=True, log_dir=None):
        self.data = data
        self.is_train = is_train
        self.log_dir = log_dir

        self.left_img_paths = utils.all_files_under(folder=os.path.join('../data', 'rg_train' + self.data),
                                                    endswith='.jpg',
                                                    condition='L_')
        self.right_img_paths = utils.all_files_under(folder=os.path.join('../data', 'rg_train' + self.data),
                                                     endswith='.jpg',
                                                     condition='R_')

        print('len of self.left_img_paths: {}'.format(len(self.left_img_paths)))
        print('len of self.right_img_paths: {}'.format(len(self.right_img_paths)))

        for left_img_path, right_img_path in zip(self.left_img_paths, self.right_img_paths):
            print('left:  {}'.format(left_img_path))
            print('right: {}'.format(right_img_path))

