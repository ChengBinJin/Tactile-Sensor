# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import argparse
from utils import all_files_under

parse = argparse.ArgumentParser(description='sepearate_data')
parse.add_argument('--folder', dest='folder', type=str, default='../data/Data', help='initial data you want to seperate')
parse.add_argument('--left_pre_fix', dest='left_pre_fix', type=str, default='A2_', help='left prefix name for the image')
parse.add_argument('--save_folder', dest='save_folder', type=str, default='../data/rg_train01', help='image saving folder')
args = parse.parse_args()

def main(read_folder, left_pre_fix, save_folder):

    file_names = all_files_under(folder=read_folder, endswith='.jpg', condition=left_pre_fix)
    total_imgs = len(file_names)

    for i, file_name in enumerate(file_names):
        left_img_name = file_name
        right_img_name = (left_img_name.replace('A', 'B')).replace('L', 'R')

        left_img = cv2.imread(left_img_name)
        right_img = cv2.imread(right_img_name)

        # Save path and restore as .png format
        save_left_path = os.path.join(save_folder, os.path.basename(left_img_name).replace('.jpg', '.png'))
        save_right_path = os.path.join(save_folder, os.path.basename(right_img_name).replace('.jpg', '.png'))

        cv2.imwrite(save_left_path, left_img)
        cv2.imwrite(save_right_path, right_img)

        if i % 200 == 0:
            print('Processing [{0:5}/{1:5}]...'.format(i, total_imgs))

if __name__ == '__main__':
    main(args.folder, args.left_pre_fix, args.save_folder)