# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import argparse
import numpy as np
import utils as utils

parser = argparse.ArgumentParser(description='')
parser.add_argument('--domain', dest='domain', type=str, default='xy', help='select data folder')
parser.add_argument('--format', dest='format', type=str, default='.jpg', help='decide image data format')
args = parser.parse_args()


def main(domain, num_attri=6):
    data_folder = os.path.join('../data', 'rg_' + domain + '_train_01')
    left_img_paths = utils.all_files_under(folder=data_folder, endswith=args.format, condition='L_')

    num_imgs = len(left_img_paths)
    data = np.zeros((num_imgs, num_attri), dtype=np.float32)
    min_max_data = np.zeros(num_attri * 2, dtype=np.float32)

    for i, img_path in enumerate(left_img_paths):
        X = float(img_path[img_path.find('_X')+2:img_path.find('_Y')])
        Y = float(img_path[img_path.find('_Y')+2:img_path.find('_Z')])
        Ra = float(img_path[img_path.find('_Ra')+3:img_path.find('_Rb')])
        Rb = float(img_path[img_path.find('_Rb')+3:img_path.find('_F')])
        F = float(img_path[img_path.find('_F')+2:img_path.find('_D')])
        D = float(img_path[img_path.find('_D')+2:img_path.find(args.format)])
        data[i, :] = np.asarray([X, Y, Ra, Rb, F, D])

    for i in range(num_attri):
        min_max_data[2*i] = data[:, i].min()
        min_max_data[2*i+1] = data[:, i].max()

    print('Min X: {}'.format(min_max_data[0]))
    print('Max X: {}'.format(min_max_data[1]))
    print('Min Y: {}'.format(min_max_data[2]))
    print('Max Y: {}'.format(min_max_data[3]))
    print('Min Ra: {}'.format(min_max_data[4]))
    print('Max Ra: {}'.format(min_max_data[5]))
    print('Min Rb: {}'.format(min_max_data[6]))
    print('Max Rb: {}'.format(min_max_data[7]))
    print('Min F: {}'.format(min_max_data[8]))
    print('Max F: {}'.format(min_max_data[9]))
    print('Min D: {}'.format(min_max_data[10]))
    print('Max D: {}'.format(min_max_data[11]))

    np.save(data_folder, min_max_data)

if __name__ == '__main__':
    main(args.domain)