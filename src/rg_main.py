# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import logging
from datetime import datetime
import tensorflow as tf

from rg_dataset import Dataset
import utils as utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('mode', 0, '0 for left-and-right input, 1 for only one camera input, default: 0')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 256')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize the original input image, default: 0.5')
tf.flags.DEFINE_string('data', '01', 'data folder name, default: 01')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, defautl: 0.0001')
tf.flags.DEFINE_integer('epoch', 1, 'number of epochs, default: 200')
tf.flags.DEFINE_integer('print_freq', 5, 'print frequence for loss information, default: 50')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20191008-151952), default: None')


def print_main_parameters(logger, flags):
    if flags.is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('mode: \t\t\t{}'.format(flags.mode))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('data: \t\t\t{}'.format(flags.data))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('weight_decay: \t\t{}'.format(flags.weight_decay))
        logger.info('epoch: \t\t\t{}'.format(flags.epoch))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t\t{}'.format(flags.gpu_index))
        print('-- mode: \t\t\t{}'.format(flags.mode))
        print('-- batch_size: \t\t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- data: \t\t\t{}'.format(flags.data))
        print('-- is_train: \t\t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t\t{}'.format(flags.weight_decay))
        print('-- epoch: \t\t\t{}'.format(flags.epoch))
        print('-- print_freq: \t\t\t{}'.format(flags.print_freq))
        print('-- load_model: \t\t\t{}'.format(flags.load_model))


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
    data = Dataset(data=FLAGS.data, is_train=FLAGS.is_train, log_dir=log_dir)


if __name__ == '__main__':
    tf.compat.v1.app.run()