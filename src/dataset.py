# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import utils as utils


class DataLoader(object):
    def __init__(self, flags, path, extension='bmp'):
        self.flags = flags
        self.img_size = (int(480 * self.flags.resize_ratio), int(640 * self.flags.resize_ratio), 3)
        self.paths = utils.all_files_under(os.path.join('../data', path), extension=extension)
        self.seed = 123  # random seed to fix random split train and validation data
        self.percentage = 0.2  # percentage used for validation data
        self.num_attributes = 7  # number of attributes for data
        self.train_paths, self.val_paths = [], []

        self._split_train_val()
        self._read_parameters()
        self._calculate_min_max_normalize()

    def _calculate_min_max_normalize(self):
        self.min_train = np.amin(self.train_data, axis=0)
        self.max_train = np.amax(self.train_data, axis=0)
        self.eps = 1e-12

        # normalize to [0, 1]
        self.norm_train_data = (self.train_data - self.min_train) / (self.max_train - self.min_train + self.eps)

        print('min_train: {}'.format(self.min_train))
        print('max_train: {}'.format(self.max_train))

    def un_normalize(self, preds):
        preds = preds * (self.max_train[0:2] - self.min_train[0:2] + self.eps) + self.min_train[0:2]
        return preds

    def _read_parameters(self):
        for idx in range(len(self.train_paths)):
            path = self.train_paths[idx]
            self.train_data[idx, 0] = float(path[path.find('_X')+2:path.find('_Y')])
            self.train_data[idx, 1] = float(path[path.find('_Y')+2:path.find('_Z')])
            self.train_data[idx, 2] = float(path[path.find('_Z')+2:path.find('_Ra')])
            self.train_data[idx, 3] = float(path[path.find('_Ra')+3:path.find('_Rb')])
            self.train_data[idx, 4] = float(path[path.find('_Rb')+3:path.find('_F')])
            self.train_data[idx, 5] = float(path[path.find('_F')+2:path.find('_D')])
            self.train_data[idx, 6] = float(path[path.find('_D')+2:path.find('.bmp')])

        for idx in range(len(self.val_paths)):
            path = self.val_paths[idx]
            self.val_data[idx, 0] = float(path[path.find('_X')+2:path.find('_Y')])
            self.val_data[idx, 1] = float(path[path.find('_Y')+2:path.find('_Z')])
            self.val_data[idx, 2] = float(path[path.find('_Z')+2:path.find('_Ra')])
            self.val_data[idx, 3] = float(path[path.find('_Ra')+3:path.find('_Rb')])
            self.val_data[idx, 4] = float(path[path.find('_Rb')+3:path.find('_F')])
            self.val_data[idx, 5] = float(path[path.find('_F')+2:path.find('_D')])
            self.val_data[idx, 6] = float(path[path.find('_D')+2:path.find('.bmp')])

    def _split_train_val(self):
        self.train_paths, self.val_paths = train_test_split(self.paths, test_size=self.percentage,
                                                            random_state=self.seed, shuffle=True)

        self.num_trains = len(self.train_paths)
        self.num_vals = len(self.val_paths)

        self.train_data = np.zeros((self.num_trains, self.num_attributes), dtype=np.float32)
        self.val_data = np.zeros((self.num_vals, self.num_attributes), dtype=np.float32)

        print('num train_paths: {}'.format(self.num_trains))
        print('num val_paths: {}'.format(self.num_vals))

    def next_batch(self):
        imgs_idx = np.random.randint(low=0, high=self.num_trains, size=self.flags.batch_size)
        imgs = [utils.load_data(self.train_paths[idx], img_size=self.img_size) for idx in imgs_idx]
        imgs = np.asarray(imgs).astype(np.float32)

        gt = []
        for idx, img_idx in enumerate(imgs_idx):
            item = np.zeros(2, dtype=np.float32)
            item[0] = self.train_data[img_idx, 0]
            item[1] = self.train_data[img_idx, 1]
            gt.append(item)
        gt_arr = np.asarray(gt)

        return imgs, gt_arr

    def next_batch_val(self):
        imgs_idx = np.random.randint(low=0, high=self.num_vals, size=self.flags.batch_size)
        imgs = [utils.load_data(self.val_paths[idx], img_size=self.img_size) for idx in imgs_idx]
        imgs = np.asarray(imgs).astype(np.float32)

        gt = []
        for idx, img_idx in enumerate(imgs_idx):
            item = np.zeros(2, dtype=np.float32)
            item[0] = self.val_data[img_idx, 0]
            item[1] = self.val_data[img_idx, 1]
            gt.append(item)
        gt_arr = np.asarray(gt)

        return imgs, gt_arr

    def test_read_img(self):
        imgs_idx = np.random.randint(low=0, high=self.num_trains, size=self.flags.batch_size)
        imgs = [utils.load_data(self.train_paths[idx], img_size=self.img_size) for idx in imgs_idx]
        imgs = np.asarray(imgs).astype(np.float32)

        for idx, img_idx in enumerate(imgs_idx):
            img = imgs[idx]
            img = img[:, :, ::-1]  # RGB to BGR
            img = img + 1. / 2.  # from [-1., 1.] to [0., 1.]

            print('sample data: {}'.format(self.train_paths[img_idx]))
            print('X: {}'.format(self.train_data[img_idx, 0]))
            print('Y: {}'.format(self.train_data[img_idx, 1]))
            print('Z: {}'.format(self.train_data[img_idx, 2]))
            print('Ra: {}'.format(self.train_data[img_idx, 3]))
            print('Rb: {}'.format(self.train_data[img_idx, 4]))
            print('F: {}'.format(self.train_data[img_idx, 5]))
            print('D: {}\n'.format(self.train_data[img_idx, 6]))

            print('Norm X: {}'.format(self.norm_train_data[img_idx, 0]))
            print('Norm Y: {}'.format(self.norm_train_data[img_idx, 1]))

            cv2.imshow('Image', img)
            cv2.waitKey(0)


def main(path):
    data_loader = DataLoader(None, path)

    random_test = np.random.randint(low=0, high=data_loader.num_vals, size=10)

    for idx in random_test:
        print(idx)
        print('sample data: {}'.format(data_loader.val_paths[idx]))
        print('X: {}'.format(data_loader.val_data[idx, 0]))
        print('Y: {}'.format(data_loader.val_data[idx, 1]))
        print('Z: {}'.format(data_loader.val_data[idx, 2]))
        print('Ra: {}'.format(data_loader.val_data[idx, 3]))
        print('Rb: {}'.format(data_loader.val_data[idx, 4]))
        print('F: {}'.format(data_loader.val_data[idx, 5]))
        print('D: {}'.format(data_loader.val_data[idx, 6]))


if __name__ == '__main__':
    data_fold_path = './data/20180908_xy'
    main(data_fold_path)

