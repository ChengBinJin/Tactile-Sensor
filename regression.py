import os
import time
import numpy as np


class DataLoader(object):
    def __init__(self, path, extension='bmp'):
        self.paths = all_files_under(path, extension=extension)
        self.X, self.Y, self.Z, self.Ra, self.Rb, self.F, self.D = [], [], [], [], [], [], []
        self.num_paths = len(self.paths)

        self.read_parameters()

    def read_parameters(self):
        print('num of data: {}'.format(self.num_paths))

        for idx in range(self.num_paths):
            path = self.paths[idx]
            self.X.append(float(path[path.find('_X')+2:path.find('_Y')]))
            self.Y.append(float(path[path.find('_Y')+2:path.find('_Z')]))
            self.Z.append(float(path[path.find('_Z')+2:path.find('_Ra')]))
            self.Ra.append(float(path[path.find('_Ra')+3:path.find('_Rb')]))
            self.Rb.append(float(path[path.find('_Rb')+3:path.find('_F')]))
            self.F.append(float(path[path.find('_F')+2:path.find('_D')]))
            self.D.append(float(path[path.find('_D')+2:path.find('.bmp')]))


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

    np.random.seed(seed=int(time.time()))
    random_test = np.random.randint(low=0, high=data_loader.num_paths, size=10)

    for idx in random_test:
        print(idx)
        print('sample data: {}'.format(data_loader.paths[idx]))
        print('X: {}'.format(data_loader.X[idx]))
        print('Y: {}'.format(data_loader.Y[idx]))
        print('Z: {}'.format(data_loader.Z[idx]))
        print('Ra: {}'.format(data_loader.Ra[idx]))
        print('Rb: {}'.format(data_loader.Rb[idx]))
        print('F: {}'.format(data_loader.F[idx]))
        print('D: {}'.format(data_loader.D[idx]))


if __name__ == '__main__':
    data_fold_path = './data/20180908_xy'
    main(data_fold_path)

