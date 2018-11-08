# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf
from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
tf.flags.DEFINE_float('resize_ratio', 0.5, 'resie ratio for originam image, default: 0.5')
tf.flags.DEFINE_integer('num_regress', 7, 'number of regresion, default: 7')

# 0: left img; 1: both imgs; 2: left circ img; 3: both circ imgs.
tf.flags.DEFINE_integer('mode', 1, 'input data type [0, 1, 2 ,3], default: 1')
tf.flags.DEFINE_string('dataset', 'train03', 'dataset name [train01, train02, train03, train04], default: train03')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate for Adam, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-5, 'hyper-parameter for regularization term, default: 0.0001')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss, default: 10')
tf.flags.DEFINE_integer('eval_freq', 500, 'evalue performance at test set, default: 500')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20180907-1739), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
