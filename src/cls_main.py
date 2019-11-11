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
import tensorflow as tf
from datetime import datetime

import utils as utils
from cls_dataset import Dataset
from cls_resnet import ResNet18
from cls_solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('mode', 0, '0 for left-and-right input, 1 for only one camera input, default: 0')
tf.flags.DEFINE_string('img_format', '.jpg', 'image format, default: .jpg')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one iteration, default: 256')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize the original input image, default: 0.5')
tf.flags.DEFINE_string('shape', 'circle', 'shape folder select from [circle|hexagon|square], default: circle')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for optimizer, default: 0.0001')
tf.flags.DEFINE_float('weight_decay', 1e-6, 'weight decay for model to handle overfitting, default: 1e-6')
tf.flags.DEFINE_integer('epoch', 1000, 'number of epochs, default: 1000')
tf.flags.DEFINE_integer('print_freq', 5, 'print frequence for loss information, default:  5')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20191110-144629), default: None')


def print_main_parameters(logger, flags):
    if flags.is_train:
        logger.info('\nmain func parameters:')
        logger.info('gpu_index: \t\t{}'.format(flags.gpu_index))
        logger.info('mode: \t\t{}'.format(flags.mode))
        logger.info('img_format: \t\t{}'.format(flags.img_format))
        logger.info('resize_factor: \t{}'.format(flags.resize_factor))
        logger.info('shape: \t\t{}'.format(flags.shape))
        logger.info('is_train: \t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t{}'.format(flags.learning_rate))
        logger.info('weight_decay: \t{}'.format(flags.weight_decay))
        logger.info('epoch: \t\t{}'.format(flags.epoch))
        logger.info('print_freq: \t\t{}'.format(flags.print_freq))
        logger.info('load_model: \t\t{}'.format(flags.load_model))
    else:
        print('main func parameters:')
        print('-- gpu_index: \t\t{}'.format(flags.gpu_index))
        print('-- mode: \t\t{}'.format(flags.mode))
        print('-- format: \t\t{}'.format(flags.img_format))
        print('-- resize_factor: \t{}'.format(flags.resize_factor))
        print('-- shape: \t\t{}'.format(flags.shape))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t{}'.format(flags.weight_decay))
        print('-- epoch: \t\t{}'.format(flags.epoch))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir = utils.make_folders_simple(cur_time=cur_time)

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, log_dir=log_dir, is_train=FLAGS.is_train, name='main')
    print_main_parameters(logger, flags=FLAGS)

    # Initialize dataset
    data = Dataset(shape=FLAGS.shape,
                   mode=FLAGS.mode,
                   img_format=FLAGS.img_format,
                   resize_factor=FLAGS.resize_factor,
                   is_train=FLAGS.is_train,
                   log_dir=log_dir,
                   is_debug=True)

    # Initialize model
    model = ResNet18(input_shape=data.input_shape,
                     num_classes=5,
                     lr=FLAGS.learning_rate,
                     weight_decay=FLAGS.weight_decay,
                     total_iters=int(np.ceil(FLAGS.epoch * data.num_train / FLAGS.batch_size)),
                     is_train=FLAGS.is_train,
                     log_dir=log_dir)

    # Initialize solver
    solver = Solver(model, data)


if __name__ == '__main__':
    tf.compat.v1.app.run()