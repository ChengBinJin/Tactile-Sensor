# ---------------------------------------------------------
# Tactile Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import _pickle as cpickle
from tensorflow.contrib.layers import flatten
import tensorflow_utils as tf_utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')


class Model(object):
    def __init__(self):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        # read pretrained weights
        weight_file_path = '../models/caffe_layers_value.pickle'
        with open(weight_file_path, 'rb') as f:
            self.pretrained_weights = cpickle.load(f, encoding='latin1')

        self._init_graph()  # initialize graph
        self.sess.run(tf.global_variables_initializer())

        # initialize model
        self.model_path = './results/train01_mode1/model/20181022-1603'
        self.saver = tf.train.Saver()
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        self.min_train = np.asarray([-10., -10., 3.14199996, 0., 0., 2.99999993e-02, 1.00000005e-03]).astype(np.float32)
        self.max_train = np.asarray([10., 10., 19.12000084, 0., 0., 0.82999998, 12.6420002]).astype(np.float32)
        self.eps = 1e-9

    def _init_graph(self):
        self.img_size = (320, 240, 2)
        self.hidden = 4096  # hyper parameters
        self.first_conv = 64
        self.resize_ratio = 0.5
        self.batch_size = 1
        self.num_regress = 7

        self.input_img_tfph = tf.placeholder(
            tf.float32, shape=[self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]], name='imgage_ph')
        self.predicts = self.network(self.input_img_tfph, name='vgg16')  # initialize graph

    def network(self, img, name=None):
        with tf.variable_scope(name):
            # conv1
            conv1_1 = tf_utils.conv2d(img, self.first_conv, k_h=3, k_w=3, d_h=1, d_w=1, name='cvon1_1')
            relu1_1 = tf_utils.relu(conv1_1, name='relu1_1')
            relu1_2 = self.conv_layer(relu1_1, 'conv1_2')
            pool_1 = tf_utils.max_pool_2x2(relu1_2, name='max_pool_1')

            # conv2
            relu2_1 = self.conv_layer(pool_1, 'conv2_1')
            relu2_2 = self.conv_layer(relu2_1, 'conv2_2')
            pool_2 = tf_utils.max_pool_2x2(relu2_2, name='max_pool_2')

            # conv3
            relu3_1 = self.conv_layer(pool_2, 'conv3_1')
            relu3_2 = self.conv_layer(relu3_1, 'conv3_2')
            relu3_3 = self.conv_layer(relu3_2, 'conv3_3')
            pool_3 = tf_utils.max_pool_2x2(relu3_3, name='max_pool_3')

            # conv4
            relu4_1 = self.conv_layer(pool_3, 'conv4_1')
            relu4_2 = self.conv_layer(relu4_1, 'conv4_2')
            relu4_3 = self.conv_layer(relu4_2, 'conv4_3')
            pool_4 = tf_utils.max_pool_2x2(relu4_3, name='max_pool_4')

            # conv5
            relu5_1 = self.conv_layer(pool_4, 'conv5_1')
            relu5_2 = self.conv_layer(relu5_1, 'conv5_2')
            relu5_3 = self.conv_layer(relu5_2, 'conv5_3')
            pool_5 = tf_utils.max_pool_2x2(relu5_3, name='max_pool_5')

            # flatten
            fc5 = flatten(pool_5)
            tf_utils.print_activations(fc5)

            # fc1
            fc6 = tf_utils.linear(fc5, self.hidden, name='fc6')
            fc6 = tf_utils.relu(fc6)
            fc6 = tf.nn.dropout(fc6, keep_prob=1.)
            tf_utils.print_activations(fc6)

            # fc2
            fc7 = self.fc_layer(fc6, 'fc7')
            fc7 = tf_utils.relu(fc7)
            fc7 = tf.nn.dropout(fc7, keep_prob=1.)
            tf_utils.print_activations(fc7)

            # fc3
            logits = tf_utils.linear(fc7, self.num_regress, name='fc8')
            tf_utils.print_activations(logits)

            return logits

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            w = self.get_conv_weight(name)
            b = self.get_bias(name)

            conv_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv_biases = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))

            conv = tf.nn.conv2d(bottom, conv_weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu_ = tf.nn.relu(bias)

            tf_utils.print_activations(relu_)

        return relu_

    def get_conv_weight(self, name):
        f = self.get_weight(name)
        return f.transpose((2, 3, 1, 0))

    def fc_layer(self, bottom, name):
        cw = self.get_weight(name)
        b = self.get_bias(name)

        if name == "fc7":
            cw = cw.transpose((1, 0))

        with tf.variable_scope(name):
            cw = tf.get_variable("W", shape=cw.shape, initializer=tf.constant_initializer(cw))
            b = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))
            fc = tf.matmul(bottom, cw) + b

        tf_utils.print_activations(fc)

        return fc

    def get_weight(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    def un_normalize(self, preds):
        preds = preds * (self.max_train - self.min_train + self.eps) + self.min_train
        return preds

    def load_model(self):
        print(' [*] Reading checkpoint...')
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_path, ckpt_name))
            return True
        else:
            return False

    def predict(self, left_img, right_img):
        # resize imgs
        left_img = cv2.resize(left_img, (int(left_img.shape[0] * self.resize_ratio),
                                         int(left_img.shape[1] * self.resize_ratio)))
        right_img = cv2.resize(right_img, (int(right_img.shape[0] * self.resize_ratio),
                                           int(right_img.shape[1] * self.resize_ratio)))
        # normalize data to [-0.5, 0.5]
        left_img = np.expand_dims(left_img / 127.5 - 1.0, axis=2).astype(np.float32)
        right_img = np.expand_dims(right_img / 127.5 - 1.0, axis=2).astype(np.float32)
        # concate left and right imgs as an input
        img = np.concatenate((left_img, right_img), axis=2)
        preds = self.sess.run(self.predicts, feed_dict={self.input_img_tfph: np.expand_dims(img, axis=0)})
        # unnormalize preds
        unnorm_preds = self.un_normalize(preds)

        return unnorm_preds


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    # read gray scale images
    left_img = cv2.imread('../data/val01/left_X0.000_Y0.000_Z3.181_Ra0.000_Rb0.000_F0.070_D0.142.bmp', 0)
    right_img = cv2.imread('../data/val01/right_X0.000_Y0.000_Z3.181_Ra0.000_Rb0.000_F0.070_D0.142.bmp', 0)

    model = Model()  # initialize model, it will take about 5 secs.

    total_pt = 0.
    num_iters = 100
    for idx in range(num_iters):
        print('[{}] / [{}]'.format(idx, num_iters))
        tic = time.time()
        preds = model.predict(left_img, right_img)
        toc = time.time() - tic
        total_pt += toc
        print('Predicts: {}'.format(preds))

    print('Avg. PT: {} ms.'.format(total_pt / num_iters * 1000))


if __name__ == '__main__':
    tf.app.run()
