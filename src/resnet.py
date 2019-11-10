# --------------------------------------------------------------------------
# Tensorflow Implementation of Tacticle Sensor Project
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import logging
import numpy as np
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils


class ResNet18_Revised(object):
    def __init__(self, input_shape, min_values, max_values, domain='xy', num_attribute=6, use_batchnorm=False, lr=1e-3,
                 weight_decay=1e-4, total_iters=2e5, small_value=1e-7, is_train=True, log_dir=None, name='ResNet18'):
        self.input_shape = input_shape
        self.min_values = min_values
        self.max_values = max_values
        self.domain = domain
        self.small_value = small_value
        self.num_attribute = num_attribute
        self.use_batchnorm=use_batchnorm
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_iters
        self.is_train = is_train
        self.log_dir = log_dir
        self.name = name
        self.layers = [2, 2, 2, 2]
        self._ops = list()
        self.tb_lr = None

        if self.domain.lower() == 'xy':
            # X, Y, Ra, Rb, F, D
            self.weights_constant = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        elif self.domain.lower() == 'rarb':
            # X, Y, Ra, Rb, F, D
            self.weights_constant = np.array([1.0, 1.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        else:
            raise NotImplementedError

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, log_dir=self.log_dir, is_train=self.is_train, name=self.name)

        self._build_graph()
        # TODO: self._best_metrics_record()
        self._eval_graph()
        self._init_tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)


    def _build_graph(self):
        self.img_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.float32, shape=[None, *self.input_shape], name='img_tfph')
        self.gt_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.float32, shape=[None, self.num_attribute], name='gt_tfph')
        self.pred_tfph = tf.compat.v1.placeholder(dtype=tf.dtypes.float32, shape=[None, self.num_attribute], name='pred_tfph')
        self.train_mode = tf.compat.v1.placeholder(dtype=tf.dtypes.bool, name='train_mode_ph')

        # Network forward for training
        self.preds = self.forward_network(input_img=self.normalize(self.img_tfph), reuse=False)
        self.unnorm_preds = self.unnormalize(self.preds)
        self.unnorm_gts = self.unnormalize(self.gt_tfph)

        # Data loss
        self.data_loss = tf.compat.v1.losses.mean_squared_error(predictions=self.weights_constant * self.preds,
                                                                labels=self.weights_constant * self.gt_tfph)
        # Regularization term
        variables = self.get_regularization_variables()
        self.reg_term = self.weight_decay * tf.math.reduce_mean([tf.nn.l2_loss(variable) for variable in variables])
        # Total loss
        self.total_loss = self.data_loss + self.reg_term

        # Optimizer
        train_op = self.init_optimizer(loss=self.total_loss)
        train_ops = [train_op] + self._ops
        self.train_op = tf.group(*train_ops)


    def _eval_graph(self):
        # Evalaution
        self.eval_ops = tf.math.reduce_mean(tf.math.sqrt(tf.math.square(self.pred_tfph - self.gt_tfph)), axis=0)
        self.avg_err = tf.math.reduce_mean(self.eval_ops)


    def _init_tensorboard(self):
        if self.is_train:
            self.tb_total = tf.compat.v1.summary.scalar('Loss/total_loss', self.total_loss)
            self.tb_data = tf.compat.v1.summary.scalar('Loss/data_loss', self.data_loss)
            self.tb_reg = tf.compat.v1.summary.scalar('Loss/reg_term', self.reg_term)
            self.summary_op = tf.compat.v1.summary.merge(
                inputs=[self.tb_total, self.tb_data, self.tb_reg, self.tb_lr])

            self.eval_summary_op = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar('Eval/X_err', self.eval_ops[0]),
                tf.compat.v1.summary.scalar('Eval/Y_err', self.eval_ops[1]),
                tf.compat.v1.summary.scalar('Eval/Ra_err', self.eval_ops[2]),
                tf.compat.v1.summary.scalar('Eval/Rb_err', self.eval_ops[3]),
                tf.compat.v1.summary.scalar('Eval/F_err', self.eval_ops[4]),
                tf.compat.v1.summary.scalar('Eval/D_err', self.eval_ops[5]),
                tf.compat.v1.summary.scalar('Eval/avg_err', self.avg_err)])


    def init_optimizer(self, loss, name='Adam'):
        with tf.compat.v1.variable_scope(name):
            global_step = tf.Variable(0., dtype=tf.float32, trainable=False)
            start_learning_rate = self.lr
            end_leanring_rate = self.lr * 0.001
            start_decay_step = int(self.total_steps * 0.5)
            decay_steps = self.total_steps - start_decay_step

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.compat.v1.train.polynomial_decay(learning_rate=start_learning_rate,
                                                                          global_step=(global_step - start_decay_step),
                                                                          decay_steps=decay_steps,
                                                                          end_learning_rate=end_leanring_rate,
                                                                          power=1.0), start_learning_rate))
            self.tb_lr = tf.compat.v1.summary.scalar('Leanring_rate', learning_rate)

            learn_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99).minimize(
                loss, global_step=global_step)

        return learn_step


    def forward_network(self, input_img, reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            tf_utils.print_activations(input_img, logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=input_img, filters=64, kernel_size=7, strides=2, name='conv1')
            inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       logger=self.logger)

            inputs = self.block_layer(inputs=inputs, filters=64, block_fn=self.bottleneck_block, blocks=self.layers[0],
                                      strides=1, train_mode=self.train_mode, name='block_layer1')
            inputs = self.block_layer(inputs=inputs, filters=128, block_fn=self.bottleneck_block, blocks=self.layers[1],
                                      strides=2, train_mode=self.train_mode, name='block_layer2')
            inputs = self.block_layer(inputs=inputs, filters=256, block_fn=self.bottleneck_block, blocks=self.layers[2],
                                      strides=2, train_mode=self.train_mode, name='block_layer3')
            inputs = self.block_layer(inputs=inputs, filters=512, block_fn=self.bottleneck_block, blocks=self.layers[3],
                                      strides=2, train_mode=self.train_mode, name='block_layer4')

            if self.use_batchnorm:
                inputs = tf_utils.norm(inputs, name='before_gap_batch_norm', _type='batch', _ops=self._ops,
                                       is_train=self.train_mode, logger=self.logger)

            inputs = tf_utils.relu(inputs, name='before_flatten_relu', logger=self.logger)

            # _, h, w, _ = inputs.get_shape().as_list()
            # inputs = tf_utils.avg_pool(inputs, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1], logger=self.logger)

            # Flatten & FC1
            inputs = tf_utils.flatten(inputs, name='flatten', logger=self.logger)
            inputs = tf_utils.linear(inputs, 512, name='FC1')
            inputs = tf_utils.relu(inputs, name='FC1_relu', logger=self.logger)

            inputs = tf_utils.linear(inputs, 256, name='FC2')
            inputs = tf_utils.relu(inputs, name='FC2_relu', logger=self.logger)

            logits = tf_utils.linear(inputs, self.num_attribute, name='Out')

            return logits


    def block_layer(self, inputs, filters, block_fn, blocks, strides, train_mode, name):
        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, train_mode, self.projection_shortcut, strides, name + '_1')

        for num_iter in range(1, blocks):
            inputs = block_fn(inputs, filters, train_mode, None, 1, name=(name + '_' + str(num_iter + 1)))

        return tf.identity(inputs, name)


    def bottleneck_block(self, inputs, filters, train_mode, projection_shortcut, strides, name):
        with tf.compat.v1.variable_scope(name):
            shortcut = inputs

            if self.use_batchnorm:
                # norm(x, name, _type, _ops, is_train=True, is_print=True, logger=None)
                inputs = tf_utils.norm(inputs, name='batch_norm_0', _type='batch', _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_0', logger=self.logger)

            # The projection shortcut shouldcome after the first batch norm and ReLU since it perofrms a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = self.projection_shortcut(inputs=inputs, filters_out=filters, strides=strides, name='conv_projection')

            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv_0')

            if self.use_batchnorm:
                inputs = tf_utils.norm(inputs, name='batch_norm_1', _type='batch', _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_1', logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_1')

            output = tf.identity(inputs + shortcut, name=(name + '_output'))
            tf_utils.print_activations(output, logger=self.logger)

            return output


    def projection_shortcut(self, inputs, filters_out, strides, name):
        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, name=name)
        return inputs


    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = tf_utils.conv2d(inputs, output_dim=filters, k_h=kernel_size, k_w=kernel_size,
                                 d_h=strides, d_w=strides, initializer='He', name=name,
                                 padding=('SAME' if strides == 1 else 'VALID'), logger=self.logger)
        return inputs


    def unnormalize(self, data):
        return data * (self.max_values - self.min_values + self.small_value) + self.min_values


    @staticmethod
    def fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
        return inputs


    @staticmethod
    def get_regularization_variables():
        # We exclude 'bias', 'beta' and 'gamma' in batch normalization
        variables = [variable for variable in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                     if ('bias' not in variable.name) and
                     ('beta' not in variable.name) and
                     ('gamma' not in variable.name)]

        return variables


    @staticmethod
    def normalize(data):
        return data - 127.5
