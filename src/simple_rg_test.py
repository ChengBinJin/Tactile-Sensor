# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from rg_dataset import Dataset
from resnet import ResNet18_Revised
from rg_solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('img_format', '.jpg', 'image format [.jpg | .png], default: .jpg')
tf.flags.DEFINE_string('domain', 'xy', 'data domain for [xy | rarb], default: xy')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize dataset
    data = Dataset(data='01' if FLAGS.domain == 'xy' else '02',
                   img_format=FLAGS.img_format,
                   is_train=False)

    # Initialize model
    model = ResNet18_Revised(input_shape=data.input_shape,
                             min_values=data.min_values,
                             max_values=data.max_values,
                             domain=FLAGS.domain,
                             is_train=False)

    # Initialize solver
    solver = Solver(model, data)

if __name__ == '__main__':
    tf.compat.v1.app.run()