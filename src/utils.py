# --------------------------------------------------------------------------
# Tensorflow Implementation of Tactile Sensor Project
# Utility functions
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import logging
import numpy as np


def make_folders_simple(cur_time=None, subfolder=''):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return model_dir, log_dir


def init_logger(logger, log_dir, name, is_train):
    logger.propagate = False  # solve print log multiple times problem
    file_handler, stream_handler = None, None

    if is_train:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    return logger, file_handler, stream_handler


def all_files_under(folder, subfolder=None, endswith='.png', condition='L_'):
    if subfolder is not None:
        new_folder = os.path.join(folder, subfolder)
    else:
        new_folder = folder

    if os.path.isdir(new_folder):
        file_names =  [os.path.join(new_folder, fname)
                       for fname in os.listdir(new_folder) if (fname.endswith(endswith)) and (condition in fname)]
        return sorted(file_names)
    else:
        return []


def read_label(img_path, img_format='.jpg'):
    X = float(img_path[img_path.find('_X') + 2:img_path.find('_Y')])
    Y = float(img_path[img_path.find('_Y') + 2:img_path.find('_Z')])
    Ra = float(img_path[img_path.find('_Ra') + 3:img_path.find('_Rb')])
    Rb = float(img_path[img_path.find('_Rb') + 3:img_path.find('_F')])
    F = float(img_path[img_path.find('_F') + 2:img_path.find('_D')])
    D = float(img_path[img_path.find('_D') + 2:img_path.find(img_format)])

    return np.asarray([X, Y, Ra, Rb, F, D])

