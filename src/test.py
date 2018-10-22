# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                       '(e.g. 20180907-1739), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    # solver = Solver(FLAGS)
    # solver.test()


if __name__ == '__main__':
    tf.app.run()
