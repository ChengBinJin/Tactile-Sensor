# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import tensorflow as tf


from dataset import DataLoader


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = DataLoader(self.flags, path=self.flags.dataset)
        self.iter_time = 0

        # self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

    def train(self):
        print('Hello train!')

        self.dataset.test_read_img()

    def test(self):
        print('Hello test!')

