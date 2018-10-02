import collections
import numpy as np
import tensorflow as tf
import _pickle as cpickle
from tensorflow.contrib.layers import flatten

import tensorflow_utils as tf_utils
import utils as utils


class VGG16_TL:
    def __init__(self, sess, flags, img_size):
        self.sess = sess
        self.flags = flags
        self.img_size = img_size
        print('self.img_size: {}'.format(self.img_size))

        self.num_regress = self.flags.num_regress
        self._extra_train_ops = []
        self.keep_prob = 0.5
        self.start_decay_step = int(np.ceil(self.flags.iters / 2))  # for optimizer
        self.decay_steps = self.flags.iters - self.start_decay_step

        # hyper_parameters
        self.hidden = 4096

        weight_file_path = '../models/caffe_layers_value.pickle'
        with open(weight_file_path, 'rb') as f:
            self.pretrained_weights = cpickle.load(f, encoding='latin1')

        self._build_model()
        self._tensorboard()

    def _build_model(self):
        self.input_img = tf.placeholder(
            tf.float32, shape=[None, self.img_size[0], self.img_size[1], self.img_size[2]], name='imgage_ph')
        self.gt_regress = tf.placeholder(tf.float32, shape=[None, self.num_regress], name='gt_ph')
        self.is_train = tf.placeholder(tf.bool, name='batch_mode_ph')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob_ph')

        self.predicts = self.network(self.input_img, self.is_train, self.keep_prob, name='vgg16')

        # data loss
        self.data_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.gt_regress, predictions=self.predicts))
        # regularization term
        self.reg_term = self.flags.weight_decay * tf.reduce_sum(
            [tf.nn.l2_loss(weight) for weight in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        # total loss
        self.total_loss = self.data_loss + self.reg_term

        optim_op = self.optimizer(loss=self.total_loss)
        train_ops = [optim_op] + self._extra_train_ops
        self.train_ops = tf.group(*train_ops)

    def _tensorboard(self):
        tf.summary.scalar('loss/reg_term', self.reg_term)
        tf.summary.scalar('loss/data_loss', self.data_loss)
        tf.summary.scalar('loss/toal_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()

    def network(self, img, mode, keep_prob, name):
        with tf.variable_scope(name):
            # conv1
            relu1_1 = self.conv_layer(img, 'conv1_1', mode)
            relu1_2 = self.conv_layer(relu1_1, 'conv1_2', mode)
            pool_1 = tf_utils.max_pool_2x2(relu1_2, name='max_pool_1')
            tf_utils.print_activations(pool_1)

            # conv2
            relu2_1 = self.conv_layer(pool_1, 'conv2_1', mode)
            relu2_2 = self.conv_layer(relu2_1, 'conv2_2', mode)
            pool_2 = tf_utils.max_pool_2x2(relu2_2, name='max_pool_2')
            tf_utils.print_activations(pool_2)

            # conv3
            relu3_1 = self.conv_layer(pool_2, 'conv3_1', mode)
            relu3_2 = self.conv_layer(relu3_1, 'conv3_2', mode)
            relu3_3 = self.conv_layer(relu3_2, 'conv3_3', mode)
            pool_3 = tf_utils.max_pool_2x2(relu3_3, name='max_pool_3')
            tf_utils.print_activations(pool_3)

            # conv4
            relu4_1 = self.conv_layer(pool_3, 'conv4_1', mode)
            relu4_2 = self.conv_layer(relu4_1, 'conv4_2', mode)
            relu4_3 = self.conv_layer(relu4_2, 'conv4_3', mode)
            pool_4 = tf_utils.max_pool_2x2(relu4_3, name='max_pool_4')
            tf_utils.print_activations(pool_4)

            # conv5
            relu5_1 = self.conv_layer(pool_4, 'conv5_1', mode)
            relu5_2 = self.conv_layer(relu5_1, 'conv5_2', mode)
            relu5_3 = self.conv_layer(relu5_2, 'conv5_3', mode)
            pool_5 = tf_utils.max_pool_2x2(relu5_3, name='max_pool_5')
            tf_utils.print_activations(pool_5)

            # flatten
            fc = flatten(pool_5)
            tf_utils.print_activations(fc)

            fc6 = tf_utils.linear(fc, self.hidden, name='fc6')
            # fc6 = tf_utils.norm(fc6, name='fc6_norm', _type='batch', _ops=self._extra_train_ops, is_train=mode)
            fc6 = tf_utils.relu(fc6)
            fc6 = tf.nn.dropout(fc6, keep_prob, name='fc6_dropout')
            tf_utils.print_activations(fc6)

            fc7 = self.fc_layer(fc6, 'fc7')
            # fc7 = tf_utils.norm(fc7, name='fc7_norm', _type='batch', _ops=self._extra_train_ops, is_train=mode)
            fc7 = tf_utils.relu(fc7)
            fc7 = tf.nn.dropout(fc7, keep_prob, name='fc6_dropout')
            tf_utils.print_activations(fc7)

            logits = tf_utils.linear(fc7, self.num_regress, name='fc8')
            logits = tf_utils.tanh(logits)
            tf_utils.print_activations(logits)

            return logits

    def train_step(self, imgs, gt_regress):
        ops = [self.train_ops, self.total_loss, self.data_loss, self.reg_term, self.summary_op]
        feed_dict = {self.input_img: imgs, self.gt_regress: gt_regress, self.is_train: True, self.keep_prob: 0.5}

        _, total_loss, data_loss, reg_term, summary = self.sess.run(ops, feed_dict=feed_dict)

        return [total_loss, data_loss, reg_term], summary

    def test_step(self, imgs):
        feed_dict = {self.input_img: imgs, self.is_train: False, self.keep_prob: 1.0}
        preds = self.sess.run(self.predicts, feed_dict=feed_dict)

        return preds

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time),
                                                  ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('toal_loss', loss[0]),
                                                  ('data_loss', loss[1]),
                                                  ('reg_term', loss[2]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def optimizer(self, loss, name='Adam'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_decay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate, power=1.0),
                                  starter_learning_rate))
        tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learn_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=name)

        return learn_step

    def conv_layer(self, bottom, name, mode):
        with tf.variable_scope(name):
            w = self.get_conv_weight(name)
            b = self.get_bias(name)

            conv_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv_biases = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))

            conv = tf.nn.conv2d(bottom, conv_weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            # batch = tf_utils.norm(bias, name='norm', _type='batch', _ops=self._extra_train_ops, is_train=mode)
            # relu_ = tf.nn.relu(batch)
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
