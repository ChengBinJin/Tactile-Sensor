import os
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, path, extension='bmp'):
        self.paths = all_files_under(path, extension=extension)
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


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def main(path):
    data_loader = DataLoader(path)

    # random_test = np.random.randint(low=0, high=data_loader.num_vals, size=10)
    #
    # for idx in random_test:
    #     print(idx)
    #     print('sample data: {}'.format(data_loader.val_paths[idx]))
    #     print('X: {}'.format(data_loader.val_data[idx, 0]))
    #     print('Y: {}'.format(data_loader.val_data[idx, 1]))
    #     print('Z: {}'.format(data_loader.val_data[idx, 2]))
    #     print('Ra: {}'.format(data_loader.val_data[idx, 3]))
    #     print('Rb: {}'.format(data_loader.val_data[idx, 4]))
    #     print('F: {}'.format(data_loader.val_data[idx, 5]))
    #     print('D: {}'.format(data_loader.val_data[idx, 6]))


if __name__ == '__main__':
    data_fold_path = './data/20180908_xy'
    main(data_fold_path)

