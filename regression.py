import os
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, path, extension='bmp'):
        self.paths = all_files_under(path, extension=extension)
        self.num_trains, self.num_vals = 0, 0

        self.seed = 123
        self.percentage = 0.2
        self.train_paths, self.val_paths = [], []
        self.train_x, self.train_y, self.train_z, self.train_ra, self.train_rb, self.train_f, self.train_d = \
            [], [], [], [], [], [], []
        self.val_x, self.val_y, self.val_z, self.val_ra, self.val_rb, self.val_f, self.val_d = \
            [], [], [], [], [], [], []

        self.split_train_val()
        self.read_parameters()

    def read_parameters(self):
        for idx in range(len(self.train_paths)):
            path = self.train_paths[idx]
            self.train_x.append(float(path[path.find('_X')+2:path.find('_Y')]))
            self.train_y.append(float(path[path.find('_Y')+2:path.find('_Z')]))
            self.train_z.append(float(path[path.find('_Z')+2:path.find('_Ra')]))
            self.train_ra.append(float(path[path.find('_Ra')+3:path.find('_Rb')]))
            self.train_rb.append(float(path[path.find('_Rb')+3:path.find('_F')]))
            self.train_f.append(float(path[path.find('_F')+2:path.find('_D')]))
            self.train_d.append(float(path[path.find('_D')+2:path.find('.bmp')]))

        for idx in range(len(self.val_paths)):
            path = self.val_paths[idx]
            self.val_x.append(float(path[path.find('_X')+2:path.find('_Y')]))
            self.val_y.append(float(path[path.find('_Y')+2:path.find('_Z')]))
            self.val_z.append(float(path[path.find('_Z')+2:path.find('_Ra')]))
            self.val_ra.append(float(path[path.find('_Ra')+3:path.find('_Rb')]))
            self.val_rb.append(float(path[path.find('_Rb')+3:path.find('_F')]))
            self.val_f.append(float(path[path.find('_F')+2:path.find('_D')]))
            self.val_d.append(float(path[path.find('_D')+2:path.find('.bmp')]))

    def split_train_val(self):
        self.train_paths, self.val_paths = train_test_split(self.paths, test_size=self.percentage,
                                                            random_state=self.seed, shuffle=True)

        self.num_trains = len(self.train_paths)
        self.num_vals = len(self.val_paths)
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

    random_test = np.random.randint(low=0, high=data_loader.num_vals, size=10)

    for idx in random_test:
        print(idx)
        print('sample data: {}'.format(data_loader.val_paths[idx]))
        print('X: {}'.format(data_loader.val_x[idx]))
        print('Y: {}'.format(data_loader.val_y[idx]))
        print('Z: {}'.format(data_loader.val_z[idx]))
        print('Ra: {}'.format(data_loader.val_ra[idx]))
        print('Rb: {}'.format(data_loader.val_rb[idx]))
        print('F: {}'.format(data_loader.val_f[idx]))
        print('D: {}'.format(data_loader.val_d[idx]))


if __name__ == '__main__':
    data_fold_path = './data/20180908_xy'
    main(data_fold_path)

