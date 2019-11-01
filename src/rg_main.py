# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import math
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf

from rg_dataset import Dataset
from rg_solver import Solver
from resnet import ResNet18_Revised
import utils as utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('mode', 0, '0 for left-and-right input, 1 for only one camera input, default: 0')
tf.flags.DEFINE_string('img_format', '.png', 'image format, default: .png')
tf.flags.DEFINE_bool('use_batchnorm', False, 'use batchnorm or not in regression task, default: False')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size for one iteration, default: 256')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize the original input image, default: 0.5')
tf.flags.DEFINE_string('data', '01', 'data folder name, default: 01')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for optimizer, default: 0.0001')
tf.flags.DEFINE_float('weight_decay', 1e-5, 'weight decay for model to handle overfitting, defautl: 0.00001')
tf.flags.DEFINE_integer('epoch', 100, 'number of epochs, default: 100')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequence for loss information, default: 5')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20191008-151952), default: None')


def print_main_parameters(logger, flags):
    if flags.is_train:
        logger.info('\nmain func parameters:')
        logger.info('gpu_index: \t\t{}'.format(flags.gpu_index))
        logger.info('mode: \t\t{}'.format(flags.mode))
        logger.info('img_format: \t\t{}'.format(flags.img_format))
        logger.info('use_batchnorm: \t{}'.format(flags.use_batchnorm))
        logger.info('batch_size: \t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t{}'.format(flags.resize_factor))
        logger.info('data: \t\t{}'.format(flags.data))
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
        print('-- use_batchnorm: \t{}'.format(flags.use_batchnorm))
        print('-- batch_size: \t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t{}'.format(flags.resize_factor))
        print('-- data: \t\t{}'.format(flags.data))
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
    data = Dataset(data=FLAGS.data,
                   mode=FLAGS.mode,
                   img_format=FLAGS.img_format,
                   resize_factor=FLAGS.resize_factor,
                   num_attribute=6,  # X, Y, Ra, Rb, F, D
                   is_train=FLAGS.is_train,
                   log_dir=log_dir,
                   is_debug=False)

    # Initialize model
    model = ResNet18_Revised(input_shape=data.input_shape,
                             min_values=data.min_values,
                             max_values=data.max_values,
                             num_attribute=data.num_atrribute,
                             use_batchnorm=FLAGS.use_batchnorm,
                             lr=FLAGS.learning_rate,
                             weight_decay=FLAGS.weight_decay,
                             total_iters=int(np.ceil(FLAGS.epoch * data.num_train / FLAGS.batch_size)),
                             is_train=FLAGS.is_train,
                             log_dir=log_dir)
    # Initialize solver
    solver = Solver(model, data)

    # Initialize saver
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    if FLAGS.is_train is True:
        train(solver, saver ,logger, model_dir, log_dir)
    else:
        test(solver, saver, model_dir, log_dir)


def train(solver, saver, logger, model_dir, log_dir):
    best_avg_err = math.inf
    iter_time, eval_time = 0, 0
    total_iters = int(np.ceil(FLAGS.epoch * solver.data.num_train / FLAGS.batch_size))
    eval_iters = total_iters // 100

    if FLAGS.load_model is not None:
        flag, iter_time = load_model(saver=saver, solver=solver, model_dir=model_dir, logger=logger, is_train=True)

        if flag is True:
            logger.info(' [!] Load Success! Iter: {}'.format(iter_time))
        else:
            exit(' [!] Failed to restore model {}'.format(FLAGS.load_model))

    # Tensorboard writer
    tb_writer = tf.compat.v1.summary.FileWriter(logdir=log_dir, graph=solver.sess.graph_def)

    while iter_time < total_iters:
        total_loss, data_loss, reg_term, summary = solver.train(batch_size=FLAGS.batch_size)

        # Print loss information
        if iter_time % FLAGS.print_freq == 0:
            msg = "[{0:6} / {1:6}] Total loss: {2:.3f}, Data loss: {3:.3f}, Reg. term: {4:.3f}"
            print(msg.format(iter_time, total_iters, total_loss, data_loss, reg_term))

            # Write to tensorboard
            tb_writer.add_summary(summary, iter_time)
            tb_writer.flush()

        if (iter_time % eval_iters == 0) or (iter_time + 1 == total_iters):
            avg_err, eval_summary = solver.eval(batch_size=FLAGS.batch_size)

            # Write the summary of evaluation on tensorboard
            tb_writer.add_summary(eval_summary, eval_time)
            tb_writer.flush()

            if avg_err < best_avg_err:
                best_avg_err = avg_err
                save_model(saver, solver, logger, model_dir, iter_time, best_avg_err)

            print('Avg. Error: {:.5f}, Best Avg. Error: {:.5f}'.format(avg_err, best_avg_err))
            eval_time += 1

        iter_time += 1


def test(solver, saver, model_dir, log_dir):
    print("Hello test!")


def save_model(saver, solver, logger, model_dir, iter_time, best_rmse):
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info('[*] Model saved: Iter: {}, Best rmse: {:.5f}'.format(iter_time, best_rmse))


def load_model(saver, solver, model_dir, logger=None, is_train=False):
    if is_train:
        logger.info(' [*] Reading checkpoint...')
    else:
        print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess, os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        return True, iter_time
    else:
        return False, None

if __name__ == '__main__':
    tf.compat.v1.app.run()